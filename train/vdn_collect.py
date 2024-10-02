"""
End-to-End JAX Implementation of VDN.

Notice:
- Agents are controlled by a single RNN architecture.
- You can choose if sharing parameters between agents or not.
- Works also with non-homogenous agents (different obs/action spaces)
- Experience replay is a simple buffer with uniform sampling.
- Uses Double Q-Learning with a target agent network (hard-updated).
- You can use TD Loss (pymarl2) or DDQN loss (pymarl)
- Adam optimizer is used instead of RMSPROP.
- The environment is reset at the end of each episode.
- Trained with a team reward (reward['__all__'])
- At the moment, last_actions are not included in the agents' observations.

The implementation closely follows the original Pymarl: https://github.com/oxwhirl/pymarl/blob/master/src/learners/q_learner.py
"""

import os
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from typing import NamedTuple, Dict, Union

import chex

import optax
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
import flashbax as fbx
import wandb
import hydra
from omegaconf import OmegaConf
from safetensors.flax import save_file
from flax.traverse_util import flatten_dict
from flax import serialization
import functools
import pickle

from jaxmarl import make
from jaxmarl.wrappers.baselines import LogWrapper, SMAXLogWrapper, CTRolloutManager
from jaxmarl.environments.smax import map_name_to_scenario
from jaxmarl.environments.overcooked import overcooked_layouts
from utils.networks import ScannedRNN, AgentRNN, batchify, unbatchify
from utils.jax_dataloader import Trajectory, Transition
from utils.tag_2_cooperative_wrapper import CooperativeEnvWrapperLoaded


def make_train(config, env):
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["NUM_ACTORS"] = config["NUM_ENVS"] * env.num_agents
    
    def train(rng):

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        wrapped_env = CTRolloutManager(env, batch_size=config["NUM_ENVS"])
        init_obs, env_state = wrapped_env.batch_reset(_rng)
        init_dones = {agent:jnp.zeros((config["NUM_ENVS"]), dtype=bool) for agent in env.agents+['__all__']}

        agent = AgentRNN(action_dim=wrapped_env.max_action_space, hidden_dim=config["AGENT_HIDDEN_DIM"], init_scale=config['AGENT_INIT_SCALE'])

        rng, _rng = jax.random.split(rng)
        init_x = (
            jnp.zeros((1, 1, wrapped_env.obs_size)), # (time_step, batch_size, obs_size)
            jnp.zeros((1, 1)) # (time_step, batch size)
        )
        init_hs = ScannedRNN.initialize_carry(1, config['AGENT_HIDDEN_DIM']) # (batch_size, hidden_dim)
        print(init_hs.shape)
        init_network_params = agent.init(_rng, init_hs, init_x)
        
        def _load_model(params):
            reward_model_path = config["EXPERT_PARAM_PATH"]
            with open(reward_model_path, "rb") as f:
                pretrain_params = serialization.from_bytes(params, f.read())
            return pretrain_params
        network_params = jax.experimental.io_callback(_load_model, init_network_params, init_network_params)
        if config.get('IDIOTIC_AGENT', False):
            def _load_idiotic_model(params):
                reward_model_path = config["IDIOTIC_AGENT_PATH"]
                print("loading idiotic model from", reward_model_path)
                with open(reward_model_path, "rb") as f:
                    pretrain_params = serialization.from_bytes(params, f.read())
                return pretrain_params
            idiotic_network_params = jax.experimental.io_callback(_load_idiotic_model, init_network_params, init_network_params)
        else:
            idiotic_network_params = network_params
      
        # TRAINING LOOP
        def _update_step(runner_state, unused):

            env_state, init_obs, init_dones, rng, timestep = runner_state

            # EPISODE STEP
            def _env_step(step_state, unused):

                params, env_state, last_obs, last_dones, hstates, rng, t = step_state
                expert_params, idiotic_params = params
                expert_hidden_state, idiotic_hidden_state = hstates
                # prepare rngs for actions and step
                rng, key_a, key_s = jax.random.split(rng, 3)

                obs_ = batchify(last_obs, env.agents, config["NUM_ACTORS"])[np.newaxis, :]
                dones_ = batchify(last_dones, env.agents, config["NUM_ACTORS"]).squeeze()[np.newaxis, :]
                expert_hidden_state, q_vals = agent.apply(expert_params, expert_hidden_state, (obs_, dones_))
                expert_actions = jnp.argmax(q_vals, axis=-1)[0]
                # change epsilon part of the actions to be random to add noises
                random_actions = jax.random.randint(key_a, (config["NUM_ACTORS"],), 0, wrapped_env.max_action_space)
                epsilon_mask = jax.random.bernoulli(key_a, config["EPSILON"], (config["NUM_ACTORS"],))
                actions = jnp.where(epsilon_mask, random_actions, expert_actions)
                
                actions = unbatchify(actions, env.agents, config["NUM_ENVS"], len(env.agents))

                # set agent_0 to be idiotic
                if config.get('IDIOTIC_AGENT', False):
                    idiotic_hidden_state, idotic_q_vals = agent.apply(idiotic_params, idiotic_hidden_state, (obs_, dones_))
                    idotic_actions = jnp.argmax(idotic_q_vals, axis=-1)[0]
                    if config['ENV_NAME'] == 'MPE_simple_tag_v3':
                        actions['adversary_0'] = unbatchify(idotic_actions, env.agents, config["NUM_ENVS"], len(env.agents))['adversary_0']
                    else:
                        actions['agent_0'] = unbatchify(idotic_actions, env.agents, config["NUM_ENVS"], len(env.agents))['agent_0']

                # STEP ENV
                obs, env_state, rewards, dones, infos = wrapped_env.batch_step(key_s, env_state, actions)
                transition = Transition(last_obs, actions, rewards, dones, infos)
                hstates = (expert_hidden_state, idiotic_hidden_state)
                step_state = (params, env_state, obs, dones, hstates, rng, t+1)
                return step_state, transition


            # prepare the step state and collect the episode trajectory
            rng, _rng = jax.random.split(rng)
            if config.get('PARAMETERS_SHARING', True):
                hstate_expert = ScannedRNN.initialize_carry(env.num_agents*config["NUM_ENVS"], config['AGENT_HIDDEN_DIM'])
                hstate_idiotic = ScannedRNN.initialize_carry(env.num_agents*config["NUM_ENVS"], config['AGENT_HIDDEN_DIM'])
                hstates = (hstate_expert, hstate_idiotic)
            else:
                hstate = ScannedRNN.initialize_carry(config['AGENT_HIDDEN_DIM'], len(env.agents), config["NUM_ENVS"]) # (n_agents, n_envs, hs_size)
            params = (network_params, idiotic_network_params)
            step_state = (
                params,
                env_state,
                init_obs,
                init_dones,
                hstates, 
                _rng,
                timestep,
            )

            step_state, traj_batch = jax.lax.scan(
                _env_step, step_state, None, config["NUM_STEPS"]
            )
            
            def callback(traj_batch, timestep):
                wandb.log(
                    {
                        "timestep": timestep,
                    } 
                )
                # if metrics['timesteps'] % 10000 == 0:
                if not os.path.exists(config["TRAJ_BATCH_PATH"]):
                    os.makedirs(config["TRAJ_BATCH_PATH"])
                file_save_path = os.path.join(config["TRAJ_BATCH_PATH"], f"traj_batch_{timestep//config['NUM_STEPS']}.pkl")
                with open(file_save_path, "wb") as f:
                    pickle.dump(traj_batch, f)
                print(f"timestep: {timestep}/{config['TOTAL_TIMESTEPS'] // config['NUM_ENVS']}, saved trajectory batch to {file_save_path}")
                if config['ENV_NAME'] == 'MPE_simple_tag_v3':
                    print(f"returns: {traj_batch.rewards['adversary_0'].sum(axis=0).mean()}")
                else:
                    print(f"returns: {traj_batch.rewards['agent_0'].sum(axis=0).mean()}")
                
                
            jax.experimental.io_callback(callback, None, traj_batch, timestep)

            runner_state = (
                env_state,
                init_obs,
                init_dones,
                rng,
                timestep + config["NUM_STEPS"],
            )

            return runner_state, None
        rng, _rng = jax.random.split(rng)
        # train
        rng, _rng = jax.random.split(rng)
        runner_state = (
            env_state,
            init_obs,
            init_dones,
            _rng,
            0,
        )
        runner_state, _ = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {'runner_state':runner_state}
    
    return train

@hydra.main(version_base=None, config_path="config", config_name="vdn_config")
def main(config):
    config = OmegaConf.to_container(config)
    config['alg']['ENV_NAME'] = config['env']['ENV_NAME']

    env_name = config["env"]["ENV_NAME"]
    alg_name = f'vdn_{"ps" if config["alg"].get("PARAMETERS_SHARING", True) else "ns"}'
    
    # smac init neeeds a scenario
    if 'smax' in env_name.lower():
        config['env']['ENV_KWARGS']['scenario'] = map_name_to_scenario(config['env']['MAP_NAME'])
        env_name = f"{config['env']['ENV_NAME']}_{config['env']['MAP_NAME']}"
        env = make(config["env"]["ENV_NAME"], **config['env']['ENV_KWARGS'])
        env = SMAXLogWrapper(env)
    # overcooked needs a layout 
    elif 'overcooked' in env_name.lower():
        config['env']["ENV_KWARGS"]["layout"] = overcooked_layouts[config['env']["ENV_KWARGS"]["layout"]]
        env = make(config["env"]["ENV_NAME"], **config['env']['ENV_KWARGS'])
        env = LogWrapper(env)
    else:
        env = make(config["env"]["ENV_NAME"], **config['env']['ENV_KWARGS'])
        env = LogWrapper(env)
        if config["env"]["ENV_NAME"] == "MPE_simple_tag_v3":
            env = CooperativeEnvWrapperLoaded(env)

    if config["alg"].get("IDIOTIC_AGENT", True):
        config["alg"]["TRAJ_BATCH_PATH"] = config["alg"]["TRAJ_BATCH_PATH"] + "unilateral"
    
    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=[
            alg_name.upper(),
            env_name.upper(),
            "RNN",
            "TD_LOSS" if config["alg"].get("TD_LAMBDA_LOSS", True) else "DQN_LOSS",
            f"jax_{jax.__version__}",
        ],
        name=f'{env_name}_collect' + str(config["SEED"]),
        config=config,
        mode=config["WANDB_MODE"],
    )
    
    rng = jax.random.PRNGKey(config["SEED"])
    train_jit = jax.jit(make_train(config["alg"], env))
    outs = train_jit(rng)

if __name__ == "__main__":
    main()
    
