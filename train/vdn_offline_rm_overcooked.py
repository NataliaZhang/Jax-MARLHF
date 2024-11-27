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
# os.environ['CUDA_VISIBLE_DEVICES'] = '5'
# os.environ['JAX_ENABLE_X64'] = 'True'
import jax
import jax.numpy as jnp
import numpy as np
import math
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
import matplotlib.animation as animation
import pickle

from jaxmarl import make
from jaxmarl.wrappers.baselines import LogWrapper, SMAXLogWrapper, CTRolloutManager
from jaxmarl.environments.mpe import MPEVisualizer
from jaxmarl.environments.mpe.simple import State
from jaxmarl.environments.smax import map_name_to_scenario
from jaxmarl.environments.overcooked import overcooked_layouts
from utils.networks import ScannedRNN, AgentRNN, ActorRNN, RewardModelFF, RewardModel, batchify, timestep_batchify, timestep_unbatchify, print_jnp_shapes
from utils.networks import ActorFF
from utils.jax_dataloader import Trajectory
from utils.tag_2_cooperative_wrapper import CooperativeEnvWrapperLoaded

class Transition_overcooked(NamedTuple):
    obs: dict
    actions: dict
    rewards: dict
    dones: dict
    infos: dict
    
class Transition(NamedTuple):
    # obs: dict
    # actions: dict
    # rewards: dict
    # dones: dict
    # infos: dict
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


def make_train(config, env):

    #load dataset
    offline_data_files = []
    for offline_data_path, num_batchs in zip(config["OFFLINE_DATA_PATHS"][config['ENV_NAME']], config["NUM_BATCH"]):
        offline_data_files.extend([offline_data_path + f"step_{i}.pkl" for i in range(num_batchs)])
    if config.get("USE_UNILATERAL", False):
        for offline_data_path, num_batchs in zip(config["OFFLINE_UNILATERAL_DATA_PATHS"][config['ENV_NAME']], config["NUM_UNILATERAL_BATCH"]):
            offline_data_files.extend([offline_data_path + f"step_{i}.pkl" for i in range(num_batchs)])
    config["OFFLINE_DATA_FILE"] = offline_data_files
    print("data ratio:", config["NUM_BATCH"])
    print("data ratio unilateral:", config["NUM_UNILATERAL_BATCH"])

    config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]

    def _load_data(idx):
        filename = config["OFFLINE_DATA_FILE"][int(idx)]
        print(f"Loading {filename}")
        with open(filename, 'rb') as f:
            traj_batch = pickle.load(f)
        traj_batch = traj_batch._replace(info=None)

        # move the loaded data to cpu
        traj_batch = jax.tree_util.tree_map(lambda x: jax.device_put(x, jax.devices('cpu')[0]), traj_batch)
        # check if traj_batch is in cpu memory
        assert jax.tree_util.tree_map(lambda x: x.device_buffer.device(), traj_batch) == jax.tree_util.tree_map(lambda x: jax.devices('cpu')[0], traj_batch)
        # print("pass the check!")

        return traj_batch
    dummy_load_traj = _load_data(0)
    
    def train(rng):
            
        # load dataset
        print("Loading dataset...")
        loaded_obs = None
        loaded_actions = None
        loaded_dones = None
        true_rewards = None
        
        def convert_data(data):
            obs = {"agent_0": data.obs[:, :16], "agent_1": data.obs[:, 16:]}
            action = {"agent_0": data.action[:, :16], "agent_1": data.action[:, 16:]}
            done = {"agent_0": data.done[:, :16], "agent_1": data.done[:, 16:]}
            reward = {"agent_0": data.reward[:, :16], "agent_1": data.reward[:, 16:]}
            world_state = obs.copy()
            return obs, action, done, reward, world_state
        
        for idx in range(len(config["OFFLINE_DATA_FILE"])):
            new_data = jax.experimental.io_callback(_load_data, dummy_load_traj, idx)
            original_traj_batch = convert_data(new_data)
            # loaded_obs = jnp.concatenate([loaded_obs, new_obs], axis=1) if loaded_obs is not None else new_obs
            # loaded_action = jnp.concatenate([loaded_action, new_action], axis=1) if loaded_action is not None else new_action
            # loaded_done = jnp.concatenate([loaded_done, new_done], axis=1) if loaded_done is not None else new_done
            # loaded_reward = jnp.concatenate([loaded_reward, new_reward], axis=1) if loaded_reward is not None else new_reward
            # loaded_world_state = jnp.concatenate([loaded_world_state, new_world_state], axis=1) if loaded_world_state is not None else new_world_state

            if loaded_obs is None: 
                loaded_obs = original_traj_batch[0]
            else: 
                loaded_obs = {k: jnp.concatenate([loaded_obs[k], original_traj_batch[0][k]], axis=1) for k in loaded_obs}
            if loaded_actions is None: 
                loaded_actions = original_traj_batch[1]
            else: 
                loaded_actions = {k: jnp.concatenate([loaded_actions[k], original_traj_batch[1][k]], axis=1) for k in loaded_actions}
            if loaded_dones is None: 
                loaded_dones = original_traj_batch[2]
            else: 
                loaded_dones = {k: jnp.concatenate([loaded_dones[k], original_traj_batch[2][k]], axis=1) for k in loaded_dones}
            if true_rewards is None: 
                true_rewards = original_traj_batch[3]
            else: 
                true_rewards = {k: jnp.concatenate([true_rewards[k], original_traj_batch[3][k]], axis=1) for k in true_rewards}


        config["DATASET_SIZE"] = loaded_obs['agent_0'].shape[1]
        print("Dataset size:", config["DATASET_SIZE"])
        print("Loaded keys and shapes of dataset:", {k: v.shape for k, v in loaded_obs.items()})

        loaded_obs = jax.tree.map(lambda x: jnp.swapaxes(x.reshape([x.shape[0], -1, config["BATCH_SIZE"]] + list(x.shape[2:])), 0, 1), loaded_obs)
        loaded_actions = jax.tree.map(lambda x: jnp.swapaxes(x.reshape([x.shape[0], -1, config["BATCH_SIZE"]] + list(x.shape[2:])), 0, 1), loaded_actions)
        loaded_dones = jax.tree.map(lambda x: jnp.swapaxes(x.reshape([x.shape[0], -1, config["BATCH_SIZE"]] + list(x.shape[2:])), 0, 1), loaded_dones)
        true_rewards = jax.tree.map(lambda x: jnp.swapaxes(x.reshape([x.shape[0], -1, config["BATCH_SIZE"]] + list(x.shape[2:])), 0, 1), true_rewards)

        print("Dataset loaded.")

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        wrapped_env = CTRolloutManager(env, batch_size=config["NUM_ENVS"], preprocess_obs=False)
        test_env = CTRolloutManager(env, batch_size=config["NUM_TEST_EPISODES"], preprocess_obs=False) # batched env for testing (has different batch size)
        init_obs, env_state = wrapped_env.batch_reset(_rng)
        init_dones = {agent:jnp.zeros((config["NUM_ENVS"]), dtype=bool) for agent in env.agents+['__all__']}

        # INIT NETWORK
        agent = AgentRNN(action_dim=6, hidden_dim=config["AGENT_HIDDEN_DIM"], init_scale=config['AGENT_INIT_SCALE'])
        rng, _rng = jax.random.split(rng)
        if config.get('PARAMETERS_SHARING', True):

            init_x = (
                jnp.zeros((1, 1, 520)), # (time_step, batch_size, obs_size)
                jnp.zeros((1, 1)) # (time_step, batch size)
            )
            init_hs = ScannedRNN.initialize_carry(1, config['AGENT_HIDDEN_DIM']) # (batch_size, hidden_dim)
            network_params = agent.init(_rng, init_hs, init_x)
        else:
            init_x = (
                jnp.zeros((env.num_agents, 1, 1, wrapped_env.obs_size)), # (time_step, batch_size, obs_size)
                jnp.zeros((env.num_agents, 1, 1)) # (time_step, batch size)
            )
            init_hs = ScannedRNN.initialize_carry(config['AGENT_HIDDEN_DIM'], env.num_agents,  1) # (n_agents, batch_size, hidden_dim)
            rngs = jax.random.split(_rng, env.num_agents) # a random init for each agent
            network_params = jax.vmap(agent.init, in_axes=(0, 0, 0))(rngs, init_hs, init_x)
            
        # INIT REWARD MODEL
        if config["REWARD_MODEL_TYPE"] == "FF":
            reward_model = RewardModelFF(action_dim=6, layer_dim=config["FF_LAYER_DIM"])
        elif config["REWARD_MODEL_TYPE"] == "RNN":
            config["RM_HIDDEN_SIZE"] = config["FF_LAYER_DIM"]
            reward_model = RewardModel(action_dim=6, hidden_size=config["RM_HIDDEN_SIZE"])
        rm_init_hstate = ScannedRNN.initialize_carry(1, config["RM_HIDDEN_SIZE"]) 
        _rm_init_rng, rng = jax.random.split(rng)
        dummy_traj = Trajectory(
            obs=jnp.zeros((1, 1, 520)),# 520 is the size of the observation space for overcooked without onehot encoding of agents
            action=jnp.ones((1, 1)),
            world_state=jnp.zeros((1, 1, 520)),
            done=jnp.zeros((1, 1)),
            reward=jnp.zeros((1, 1)),
            log_prob=None,
            avail_actions=None,
        )
        _rm_init_x = dummy_traj
        rm_params = reward_model.init(_rm_init_rng, rm_init_hstate, _rm_init_x)
        def _load_rm_model(params):
            with open(rm_model_path, "rb") as f:
                pretrain_params = serialization.from_bytes(params, f.read())
            print("Loaded reward model params from ", rm_model_path)
            print("...")
            return pretrain_params
        
        rm_model_path = os.path.join(config["REWARD_NETWORK_PARAM_PATH"], "reward_model_best.msgpack")
        rm_params = jax.experimental.io_callback(_load_rm_model, rm_params, rm_params)

        # INIT REFERENCE AGENT (learnt with imitation learning)
        reference_model = ActorFF(action_dim=wrapped_env.max_action_space, config=config)
        # re_init_hstate = ScannedRNN.initialize_carry(1, config["RE_HIDDEN_SIZE"])
        # _re_init_x = (
        #     jnp.zeros((1, 1, 520)),
        #     jnp.zeros((1, 1))
        # )
        _s_init_rng, rng = jax.random.split(rng)
        init_x = jnp.zeros(env.observation_space().shape)
        
        init_x = init_x.flatten()
        re_params = reference_model.init(_s_init_rng, init_x)

        # use the first batch to compute standarize the predicted rewards
        get_first = lambda x: x[0]
        _obs, _action, _done = *(jax.tree.map(get_first, (loaded_obs, loaded_actions,loaded_dones))),
        obs_batch = jnp.concatenate([_obs[a] for a in env.agents], axis=1) # (time_steps, n_actors, obs_size)
        # obs_batch = jnp.concatenate([_obs[a] for a in env.agents], axis=1)[..., :-env.num_agents] # (time_steps, n_actors, obs_size)
        action_batch = jnp.concatenate([_action[a] for a in env.agents], axis=1)
        done_batch = jnp.concatenate([_done[a] for a in env.agents], axis=1)
        world_state = obs_batch.copy()
        # world_state = _obs['__all__'].repeat(env.num_agents, axis=1)
        
        #calculate the reward
        traj_to_predict = Trajectory(
            obs=obs_batch, # remove the one-hot encoding of agents
            action=action_batch,
            world_state=world_state, 
            done=done_batch,
        ) 
        _, predicted_reward = reward_model.apply(rm_params, ScannedRNN.initialize_carry(obs_batch.shape[1], config["RM_HIDDEN_SIZE"]), traj_to_predict)
        batched_reward = predicted_reward.reshape([obs_batch.shape[0], env.num_agents, config["BATCH_SIZE"]]) # (time_steps, n_agents, batch_size)
        config["REWARD_STD"] = batched_reward.std()
        config["REWARD_MEAN"] = batched_reward.mean()


        def _load_ref_model(params):
            with open(ref_model_path, "rb") as f:
                pretrain_params = serialization.from_bytes(params, f.read())
            print("Loaded reference model params from ", ref_model_path)
            return pretrain_params
        
        # ref_model_path = config["REFERENCE_NETWORK_PARAM_PATH"]
        ref_model_path = config["REFERENCE_NETWORK_PARAM_PATH"] + "|".join([str(r) for r in config["NUM_BATCH"]]) +'_' +"|".join([str(r) for r in config["NUM_UNILATERAL_BATCH"]]) + '.msgpack'
        re_params = jax.experimental.io_callback(_load_ref_model, re_params, re_params)
            
        # INIT OPTIMIZER
        def linear_schedule(count):
            frac = 1.0 - (count / (config["EPOCHS"]))
            return config["LR"] * frac
        lr = linear_schedule if config.get('LR_LINEAR_DECAY', False) else config['LR']
        tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adam(learning_rate=lr, eps=config['EPS_ADAM']),
        )
        train_state = TrainState.create(
            apply_fn=agent.apply,
            params=network_params,
            tx=tx,
        )
        # target network params
        target_agent_params = jax.tree.map(lambda x: jnp.copy(x), train_state.params)

        # depending if using parameters sharing or not, q-values are computed using one or multiple parameters
        if config.get('PARAMETERS_SHARING', True):
            def homogeneous_pass(params, hidden_state, obs, dones):
                # concatenate agents and parallel envs to process them in one batch
                agents, flatten_agents_obs = zip(*obs.items())
                original_shape = flatten_agents_obs[0].shape # assumes obs shape is the same for all agents
                
                batched_input = (
                    jnp.concatenate(flatten_agents_obs, axis=1), # (time_step, n_agents*n_envs, obs_size)
                    jnp.concatenate([dones[agent] for agent in agents], axis=1), # ensure to not pass other keys (like __all__)
                )
                print("shape of elements in batched_input", [x.shape for x in batched_input])
                hidden_state, q_vals = agent.apply(params, hidden_state, batched_input) #(time_step, n_actors, action_dim)
                q_vals = q_vals.reshape(original_shape[0], len(agents), *original_shape[1:-1], -1) # (time_steps, n_agents, n_envs, action_dim)
                q_vals = {a:q_vals[:,i] for i,a in enumerate(agents)}
                return hidden_state, q_vals
        else:
            def homogeneous_pass(params, hidden_state, obs, dones):
                # homogeneous pass vmapped in respect to the agents parameters (i.e., no parameter sharing)
                agents, flatten_agents_obs = zip(*obs.items())
                batched_input = (
                    jnp.stack(flatten_agents_obs, axis=0), # (n_agents, time_step, n_envs, obs_size)
                    jnp.stack([dones[agent] for agent in agents], axis=0), # ensure to not pass other keys (like __all__)
                )
                # computes the q_vals with the params of each agent separately by vmapping
                hidden_state, q_vals = jax.vmap(agent.apply, in_axes=0)(params, hidden_state, batched_input)
                q_vals = {a:q_vals[i] for i,a in enumerate(agents)}
                return hidden_state, q_vals

        # add rewards to the loaded dataset with the reward model
        
        def _add_rewards_log_prob(not_used, obs_a_d):
            _obs, _action, _done= obs_a_d
            obs_batch = jnp.concatenate([_obs[a] for a in env.agents], axis=1)# (time_steps, n_actors, obs_size)
            # obs_batch = jnp.concatenate([_obs[a] for a in env.agents], axis=1)[..., :-env.num_agents] # (time_steps, n_actors, obs_size)
            action_batch = jnp.concatenate([_action[a] for a in env.agents], axis=1)
            done_batch = jnp.concatenate([_done[a] for a in env.agents], axis=1)
            world_state = obs_batch.copy()
            
            # calculate the prediected reward
            traj_to_predict = Trajectory(
                obs=obs_batch, # removed the one-hot encoding of agents
                action=action_batch,
                world_state=world_state, 
                done=done_batch,
            ) 
            _, predicted_reward = reward_model.apply(rm_params, ScannedRNN.initialize_carry(obs_batch.shape[1], config["RM_HIDDEN_SIZE"]), traj_to_predict)
            predicted_reward = (predicted_reward - config["REWARD_MEAN"]) / config["REWARD_STD"] + 1
            batched_reward = predicted_reward.reshape([obs_batch.shape[0], env.num_agents, config["BATCH_SIZE"]]) # (time_steps, n_agents, batch_size)
            unbatched_reward = {a:batched_reward[:, i] for i, a in enumerate(env.agents)}
            unbatched_reward['__all__'] = jnp.array([v for k, v in unbatched_reward.items()]).sum(axis=0)
            
            # calculate the log_prob
            re_x = (obs_batch, done_batch)
            # _, reference_pi = reference_model.apply(re_params, ScannedRNN.initialize_carry(obs_batch.shape[1], config["RE_HIDDEN_SIZE"]), re_x)
            reference_pi = reference_model.apply(re_params, obs_batch)
            
            batched_log_prob = reference_pi.log_prob(action_batch)
            batched_reward_with_log = predicted_reward + jnp.clip(config["REF_LOG_COEF"] * (batched_log_prob + jnp.log(wrapped_env.max_action_space)), -10.0, 1.0)
            # batched_reward_with_log = jnp.clip(config["REF_LOG_COEF"] * (batched_log_prob + jnp.log(wrapped_env.max_action_space)), -10.0, 1.0) # TODO: change back to the previous line!
            batched_reward_with_log = batched_reward_with_log.reshape([obs_batch.shape[0], env.num_agents, config["BATCH_SIZE"]]) # (time_steps, n_agents, batch_size)
            unbatched_reward_with_log = {a:batched_reward_with_log[:, i] for i, a in enumerate(env.agents)}
            unbatched_reward_with_log['__all__'] = jnp.array([v for k, v in unbatched_reward_with_log.items()]).sum(axis=0)
            
            return None, (unbatched_reward, unbatched_reward_with_log)
        
        _, rewards_log_prob = jax.lax.scan(_add_rewards_log_prob, None, (loaded_obs, loaded_actions, loaded_dones))
        loaded_rewards, loaded_rewards_with_log = rewards_log_prob

        if config["ADD_LOG_PROB"]:
            dataset = Transition_overcooked(
                obs=loaded_obs,
                actions=loaded_actions,
                rewards=loaded_rewards_with_log,
                dones=loaded_dones,
                infos=None)
        else:
            dataset = Transition_overcooked(
                obs=loaded_obs,
                actions=loaded_actions,
                rewards=loaded_rewards,
                dones=loaded_dones,
                infos=None)
        dataset = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1).reshape([x.shape[1], -1] + list(x.shape[3:])), dataset)
        print("Loaded keys and shapes of dataset:", {k: v.shape for k, v in dataset.obs.items()})
        
        # INIT BUFFER
        # to initalize the buffer is necessary to sample a trajectory to know its strucutre
        sample_traj_unbatched = jax.tree.map(lambda x: x[:, 0], dataset) # remove the NUM_ENV dim
        buffer = fbx.make_trajectory_buffer(
            max_length_time_axis=config['BUFFER_SIZE']//config['NUM_ENVS'],
            min_length_time_axis=config['BUFFER_BATCH_SIZE'],
            sample_batch_size=config['BUFFER_BATCH_SIZE'],
            add_batch_size=config['DATASET_SIZE'],
            sample_sequence_length=1,
            period=1,
        )
        buffer_state = buffer.init(sample_traj_unbatched) 
        
        # BUFFER UPDATE
        # save the collected trajectory in the buffer
        buffer_traj_batch = jax.tree_util.tree_map(
            lambda x:jnp.swapaxes(x, 0, 1)[:, np.newaxis], # put the batch dim first and add a dummy sequence dim
            dataset
        ) # (num_envs, 1, time_steps, ...)
        buffer_state = buffer.add(buffer_state, buffer_traj_batch)
        # move buffer to gpu
        # buffer_state = jax.tree.map(lambda x: jax.device_put(x, device=jax.devices('gpu')[0]), buffer_state)
        # print("Buffer state obs shape:", {k: v.shape for k, v in buffer_state['experience'].obs.items()})
        # print("Buffer state rewards shape:", {k: v.shape for k, v in buffer_state['experience'].rewards.items()})
        
        # TRAINING LOOP
        def _update_step(runner_state, epoch):

            train_state, target_agent_params, env_state, buffer_state, time_state, init_obs, init_dones, test_metrics, rng = runner_state

            # LEARN PHASE
            def q_of_action(q, u):
                """index the q_values with action indices"""
                q_u = jnp.take_along_axis(q, jnp.expand_dims(u, axis=-1), axis=-1)
                return jnp.squeeze(q_u, axis=-1)

            def _loss_fn(params, target_agent_params, init_hs, learn_traj):
                learn_traj.dones['__all__'] = learn_traj.dones['agent_0'].copy() # set the global done key
                learn_traj.rewards['__all__'] = jnp.array([v for k, v in learn_traj.rewards.items()]).sum(axis=0) # set the global reward key

                obs_ = {a:learn_traj.obs[a] for a in env.agents} # ensure to not pass the global state (obs["__all__"]) to the network
                _, q_vals = homogeneous_pass(params, init_hs, obs_, learn_traj.dones)
                _, target_q_vals = homogeneous_pass(target_agent_params, init_hs, obs_, learn_traj.dones)

                # get the q_vals of the taken actions (with exploration) for each agent
                chosen_action_qvals = jax.tree.map(
                    lambda q, u: q_of_action(q, u.squeeze())[:-1], # avoid last timestep
                    q_vals,
                    learn_traj.actions
                )

                # get the target q values of the greedy actions
                # valid_q_vals = jax.tree_util.tree_map(lambda q, valid_idx: q[..., valid_idx], q_vals, wrapped_env.valid_actions)
                target_max_qvals = jax.tree.map(
                    lambda t_q, q: q_of_action(t_q, jnp.argmax(q, axis=-1))[1:], # get the greedy actions and avoid first timestep
                    target_q_vals,
                    q_vals
                )

                # VDN: computes q_tot as the sum of the agents' individual q values
                chosen_action_qvals_sum = jnp.stack(list(chosen_action_qvals.values())).sum(axis=0)
                target_max_qvals_sum = jnp.stack(list(target_max_qvals.values())).sum(axis=0)

                # compute the centralized targets using the "__all__" rewards and dones
                if config.get('TD_LAMBDA_LOSS', True):
                    # time difference loss
                    def _td_lambda_target(ret, values):
                        reward, done, target_qs = values
                        ret = jnp.where(
                            done,
                            # target_qs,
                            reward,
                            ret*config['TD_LAMBDA']*config['GAMMA']
                            + reward
                            + (1-config['TD_LAMBDA'])*config['GAMMA']*target_qs
                        )
                        return ret, ret

                    ret = target_max_qvals_sum[-1] * (1-learn_traj.dones['__all__'][-1])
                    ret, td_targets = jax.lax.scan(
                        _td_lambda_target,
                        ret,
                        (learn_traj.rewards['__all__'][-2::-1], learn_traj.dones['__all__'][-2::-1], target_max_qvals_sum[-1::-1])
                    )
                    targets = td_targets[::-1]
                    loss = jnp.mean(0.5*((chosen_action_qvals_sum - jax.lax.stop_gradient(targets))**2))
                else:
                    # standard DQN loss
                    targets = (
                        learn_traj.rewards['__all__'][:-1]
                        + config['GAMMA']*(1-learn_traj.dones['__all__'][:-1])*target_max_qvals_sum
                    )
                    loss = jnp.mean((chosen_action_qvals_sum - jax.lax.stop_gradient(targets))**2)

                return loss


            # sample a batched trajectory from the buffer and set the time step dim in first axis
            rng, _rng = jax.random.split(rng)
            learn_traj = buffer.sample(buffer_state, _rng).experience # (batch_size, 1, max_time_steps, ...)
            # print("learn_traj shape", learn_traj.obs["__all__"].shape)

            # # move the learn_traj to gpu
            # learn_traj = jax.tree_util.tree_map(lambda x: jax.device_put(x, jax.devices('gpu')[0]), learn_traj)

            learn_traj = jax.tree.map(
                lambda x: jnp.swapaxes(x[:, 0], 0, 1), # remove the dummy sequence dim (1) and swap batch and temporal dims
                learn_traj
            ) # (max_time_steps, batch_size, ...)
            if config.get('PARAMETERS_SHARING', True):
                init_hs = ScannedRNN.initialize_carry(env.num_agents*config["BUFFER_BATCH_SIZE"], config['AGENT_HIDDEN_DIM']) # (n_agents*batch_size, hs_size)
            else:
                init_hs = ScannedRNN.initialize_carry(config['AGENT_HIDDEN_DIM'], env.num_agents, config["BUFFER_BATCH_SIZE"]) # (n_agents, batch_size, hs_size)

            # compute loss and optimize grad
            grad_fn = jax.value_and_grad(_loss_fn, has_aux=False)
            loss, grads = grad_fn(train_state.params, target_agent_params, init_hs, learn_traj)
            train_state = train_state.apply_gradients(grads=grads)


            # UPDATE THE VARIABLES AND RETURN
            # reset the environment
            rng, _rng = jax.random.split(rng)
            init_obs, env_state = wrapped_env.batch_reset(_rng)
            init_dones = {agent:jnp.zeros((config["NUM_ENVS"]), dtype=bool) for agent in env.agents+['__all__']}

            # update the states
            time_state['timesteps'] = time_state['timesteps'] + config['NUM_STEPS']
            time_state['updates']   = time_state['updates'] + 1

            # update the target network if necessary
            sqrt_updates = jnp.sqrt(time_state['updates'])
            # condition = (sqrt_updates % config['TARGET_UPDATE_INTERVAL']) == 0.0
            condition = time_state['updates'] % config['TARGET_UPDATE_INTERVAL'] == 0
            target_agent_params = jax.lax.cond(
                condition,
                lambda _: jax.tree.map(lambda x: jnp.copy(x), train_state.params),
                lambda _: target_agent_params,
                operand=None
            )

            # update the greedy rewards
            rng, _rng = jax.random.split(rng)
            test_metrics = jax.lax.cond(
                time_state['updates'] % (config["TEST_INTERVAL"] // config["NUM_STEPS"] // config["NUM_ENVS"]) == 0,
                lambda _: get_greedy_metrics(_rng, train_state.params, time_state, epoch),
                lambda _: test_metrics,
                operand=None
            )

            # update the returning metrics
            metrics = {
                'timesteps': time_state['timesteps'],
                'updates' : time_state['updates'],
                'loss': loss,
                'rewards': jax.tree_util.tree_map(lambda x: jnp.sum(x, axis=0).mean(), learn_traj.rewards), # learn_traj.rewards['__all__']: (max_time_steps, batch_size)
                'test_metrics': test_metrics  # add the test metrics dictionary
            }
            # compute the metrics of the first episode that is done for each parallel env
            # TODO: merge it with the same fucntion used in get_greedy_metrics
            def first_episode_returns(rewards, dones):
                first_done = jax.lax.select(jnp.argmax(dones)==0., dones.size, jnp.argmax(dones))
                first_episode_mask = jnp.where(jnp.arange(dones.size) <= first_done, True, False)
                return jnp.where(first_episode_mask, rewards, 0.).sum()
            first_returns = jax.tree.map(lambda r: jax.vmap(first_episode_returns, in_axes=1)(r, learn_traj.dones["__all__"]), learn_traj.rewards["__all__"])
            metrics['first_returns'] = first_returns.mean() / env.num_agents

            # if config.get('WANDB_ONLINE_REPORT', False):
            def callback(metrics, epoch):
                wandb.log(
                    {
                        "training_returns": metrics['first_returns'],
                        "loss": metrics['loss'],
                    },
                    step=int(epoch),
                )
                # if metrics['timesteps'] % 10000 == 0:
                print(f"train_timestep: {metrics['timesteps'] * config['NUM_ENVS']}, loss: {metrics['loss']}, training_returns: {metrics['first_returns']}")
            jax.experimental.io_callback(callback, None, metrics, epoch)
            # jax.debug.callback(callback, metrics, traj_batch.infos)

            runner_state = (
                train_state,
                target_agent_params,
                env_state,
                buffer_state,
                time_state,
                init_obs,
                init_dones,
                test_metrics,
                rng
            )

            return runner_state, metrics

        def get_greedy_metrics(rng, params, time_state, epoch):
            """Help function to test greedy policy during training"""
            def _greedy_env_step(step_state, unused):
                params, env_state, last_obs, last_dones, hstate, rng = step_state
                rng, key_s = jax.random.split(rng)
                obs_   = {a:last_obs[a] for a in env.agents}
                # obs_  = obs_.reshape([1, obs_.shape[0], -1])
                obs_  = jax.tree.map(lambda x: x.reshape([1, x.shape[0], -1]), obs_)
                
                # obs_   = jax.tree.map(lambda x: x[np.newaxis, :], obs_)
                dones_ = jax.tree.map(lambda x: x[np.newaxis, :], last_dones)
                hstate, q_vals = homogeneous_pass(params, hstate, obs_, dones_)
                actions = jax.tree_util.tree_map(lambda q: jnp.argmax(q.squeeze(0), axis=-1), q_vals)
                obs, env_state, rewards, dones, infos = test_env.batch_step(key_s, env_state, actions)
                
                # calculate predicted rewards using the reward model
                obs_batch = jnp.concatenate([obs_[a] for a in env.agents], axis=1) # (time_steps, n_actors, obs_size)
                # obs_batch = jnp.concatenate([obs_[a] for a in env.agents], axis=1)[..., :-env.num_agents] # (time_steps, n_actors, obs_size)
                action_batch = jnp.concatenate([actions[a] for a in env.agents], axis=0).reshape([1, -1]) # (time_steps, n_actors)
                dones_batch = jnp.concatenate([dones_[a] for a in env.agents], axis=1)
                # world_state = obs_batch.repeat(env.num_agents, axis=-1)
                world_state = obs_batch.copy()
                traj_to_predict = Trajectory(
                    obs=obs_batch, # remove the one-hot encoding of agents
                    action=action_batch,
                    world_state=world_state, 
                    done=dones_batch
                )
                
                # print("obs_batch shape", obs_batch.shape)
                # print("config['RM_HIDDEN_SIZE']", config["RM_HIDDEN_SIZE"])
                # print("traj_to_predict.obs shape", traj_to_predict.obs.shape)
                
                _, predicted_rewards = reward_model.apply(rm_params, ScannedRNN.initialize_carry(obs_batch.shape[1], config["RM_HIDDEN_SIZE"]), traj_to_predict)
                predicted_rewards = (predicted_rewards - config["REWARD_MEAN"]) / config["REWARD_STD"] + 1
                batched_predicted_rewards = predicted_rewards.reshape([obs_batch.shape[0], env.num_agents, -1])    # (time_steps, n_agents, batch_size)
                unbatched_predicted_rewards = {a: batched_predicted_rewards[:, i].squeeze(0) for i, a in enumerate(env.agents)}
                unbatched_predicted_rewards['__all__'] = jnp.array([v for k, v in unbatched_predicted_rewards.items()]).sum(axis=0)

                # calculate the log_prob
                re_x = (obs_batch, dones_batch)
                # _, reference_pi = reference_model.apply(re_params, ScannedRNN.initialize_carry(obs_batch.shape[1], config["RE_HIDDEN_SIZE"]), re_x)
                reference_pi = reference_model.apply(re_params, obs_batch)
                batched_log_prob = reference_pi.log_prob(action_batch)
                batched_reward_with_log = predicted_rewards + jnp.clip(config["REF_LOG_COEF"] * (batched_log_prob + jnp.log(wrapped_env.max_action_space)), -10.0, 1.0)
                # batched_reward_with_log = jnp.clip(config["REF_LOG_COEF"] * (batched_log_prob + jnp.log(wrapped_env.max_action_space)), -10.0, 1.0) # TODO: change back to the previous line!
                batched_reward_with_log = batched_reward_with_log.reshape([obs_batch.shape[0], env.num_agents, -1]) # (time_steps, n_agents, batch_size)
                unbatched_reward_with_log = {a:batched_reward_with_log[:, i].squeeze(0) for i, a in enumerate(env.agents)}
                unbatched_reward_with_log['__all__'] = jnp.array([v for k, v in unbatched_reward_with_log.items()]).sum(axis=0)

                step_state = (params, env_state, obs, dones, hstate, rng)
                return step_state, (rewards, dones, infos, unbatched_predicted_rewards, unbatched_reward_with_log)
            
            rng, _rng = jax.random.split(rng)
            init_obs, env_state = test_env.batch_reset(_rng)
            init_dones = {agent:jnp.zeros((config["NUM_TEST_EPISODES"]), dtype=bool) for agent in env.agents+['__all__']}
            rng, _rng = jax.random.split(rng)
            if config["PARAMETERS_SHARING"]:
                hstate = ScannedRNN.initialize_carry(env.num_agents*config["NUM_TEST_EPISODES"], config['AGENT_HIDDEN_DIM']) # (n_agents*n_envs, hs_size)
            else:
                hstate = ScannedRNN.initialize_carry(config['AGENT_HIDDEN_DIM'], env.num_agents, config["NUM_TEST_EPISODES"]) # (n_agents, n_envs, hs_size)
            step_state = (
                params,
                env_state,
                init_obs,
                init_dones,
                hstate, 
                _rng,
            )
            step_state, (rewards, dones, infos, predicted_rewards, rewards_with_log) = jax.lax.scan(
                _greedy_env_step, step_state, None, config["NUM_STEPS"]
            )
            # compute the metrics of the first episode that is done for each parallel env
            def first_episode_returns(rewards, dones):
                first_done = jax.lax.select(jnp.argmax(dones)==0., dones.size, jnp.argmax(dones))
                first_episode_mask = jnp.where(jnp.arange(dones.size) <= first_done, True, False)
                return jnp.where(first_episode_mask, rewards, 0.).sum()
            
            all_dones = dones['__all__']
            first_returns = jax.tree.map(lambda r: jax.vmap(first_episode_returns, in_axes=1)(r, all_dones), rewards)
            first_infos   = jax.tree.map(lambda i: jax.vmap(first_episode_returns, in_axes=1)(i[..., 0], all_dones), infos)
            first_predicted_rewards = jax.tree.map(lambda r: jax.vmap(first_episode_returns, in_axes=1)(r, all_dones), predicted_rewards)
            first_rewards_with_log = jax.tree.map(lambda r: jax.vmap(first_episode_returns, in_axes=1)(r, all_dones), rewards_with_log)

            metrics = {
                'test_returns': first_returns['__all__'],   # episode returns
                'test_predicted_rewards': first_predicted_rewards['__all__'],
                **{'test_'+k:v for k,v in first_infos.items()}
            }
            if config.get('VERBOSE', False):
                def callback(epoch, test_returns, predicted_returns, predicted_returns_with_log, train_state):
                    wandb.log(
                        {
                            "test_returns": test_returns,
                            "test_predicted_rewards": predicted_returns,
                            "test_predicted_rewards_with_log": predicted_returns_with_log,
                            "test_reward_reference_log_term": predicted_returns_with_log - predicted_returns,
                            }, 
                        step=int(epoch),
                    )
                    print(f"Epoch: {epoch}, test return: {test_returns}, test predicted rewards: {predicted_returns}, test predicted rewards with log: {predicted_returns_with_log}")

                    if epoch == config["EPOCHS"] - 1:
                        # save params              
                        os.makedirs(config["SAVE_PATH"], exist_ok=True)
                        with open(f'{config["SAVE_PATH"]}/vdn_model.safetensors', 'wb') as f:
                            f.write(serialization.to_bytes(train_state.params))
                        print(f'Parameters of first batch saved in {config["SAVE_PATH"]}/vdn_model.safetensors')

                jax.debug.callback(callback, epoch, first_returns['__all__'].mean(), first_predicted_rewards['__all__'].mean(), first_rewards_with_log['__all__'].mean(), train_state)
            return metrics

        time_state = {
            'timesteps':jnp.array(0),
            'updates':  jnp.array(0)
        }
        rng, _rng = jax.random.split(rng)
        test_metrics = get_greedy_metrics(_rng, train_state.params, time_state, 0) # initial greedy metrics
        
        # train
        rng, _rng = jax.random.split(rng)
        runner_state = (
            train_state,
            target_agent_params,
            env_state,
            buffer_state,
            time_state,
            init_obs,
            init_dones,
            test_metrics,
            _rng
        )
        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, np.arange(config["EPOCHS"])
        )
        return {'runner_state': runner_state, 'metrics': metrics}
    
    return train

@hydra.main(version_base=None, config_path="config", config_name="vdn_config_overcooked")
def main(config):
    config = OmegaConf.to_container(config)
    config['alg']['ENV_NAME'] = config['env']['ENV_NAME']
    # print('Config:\n', OmegaConf.to_yaml(config))

    env_name = config["env"]["ENV_NAME"]
    alg_name = f'vdn_{"ps" if config["alg"].get("PARAMETERS_SHARING", True) else "ns"}'
    if env_name.split("_")[0] == 'MPE':
        env_name_str = env_name.split("_")[2]
    else:
        env_name_str = env_name

    # if config["alg"].get("USE_UNILATERAL", False):
    config["alg"]["REWARD_NETWORK_PARAM_PATH"] = f'results/{env_name_str}/unilateral/rm_{config["alg"]["REWARD_MODEL_TYPE"]}_{config["alg"]["FF_LAYER_DIM"]}_mse_{config["alg"]["MSE_LOSS_COEF"]}_vdn_' + '|'.join([str(x) for x in config["alg"]['NUM_BATCH']]) + "_" + '|'.join([str(x) for x in config["alg"]['NUM_UNILATERAL_BATCH']])
    # else:
    # config["alg"]["REWARD_NETWORK_PARAM_PATH"] = f'results/{env_name_str}/rm_{config["alg"]["REWARD_MODEL_TYPE"]}_mse_{config["alg"]["MSE_LOSS_COEF"]}_vdn_' + '|'.join([str(x) for x in config["alg"]['NUM_BATCH']])
    
    config["alg"]["SAVE_PATH"] = f'results/vdn/{env_name_str}/vdn_rm_{config["alg"]["REWARD_MODEL_TYPE"]}_{config["alg"]["FF_LAYER_DIM"]}_mse_{config["alg"]["MSE_LOSS_COEF"]}_vdn_' + '|'.join([str(x) for x in config["alg"]['NUM_BATCH']]) + "_" + '|'.join([str(x) for x in config["alg"]['NUM_UNILATERAL_BATCH']])

    # smac init neeeds a scenario
    # if 'smax' in env_name.lower():
    #     config['env']['ENV_KWARGS']['scenario'] = map_name_to_scenario(config['env']['MAP_NAME'])
    #     env_name = f"{config['env']['ENV_NAME']}_{config['env']['MAP_NAME']}"
    #     env = make(config["env"]["ENV_NAME"], **config['env']['ENV_KWARGS'])
    #     env = SMAXLogWrapper(env)
    # overcooked needs a layout 
    # elif 'overcooked' in env_name.lower():
    config['env']['ENV_KWARGS']["layout"] = overcooked_layouts[config['env']['ENV_KWARGS']["layout"]]
    env = make(config["env"]['ENV_NAME'], **config['env']['ENV_KWARGS'])
    env = LogWrapper(env)
    # else:
    #     if config['env']['ENV_NAME'] == "MPE_simple_spread_scale":
    #         config['env']['ENV_KWARGS'] = {
    #             "num_agents": config['env']["NUM_AGENTS"], 
    #             "num_landmarks": config['env']["NUM_LANDMARKS"]
    #         }
    #     env = make(config['env']['ENV_NAME'], **config['env']['ENV_KWARGS'])
    #     env = LogWrapper(env)
    #     if config['env']['ENV_NAME'] == "MPE_simple_tag_v3":
    #         env = CooperativeEnvWrapperLoaded(env)

    #config["alg"]["NUM_STEPS"] = config["alg"].get("NUM_STEPS", env.max_steps) # default steps defined by the env
    
    wandb_name = f'{alg_name}_{env_name}_offline_mse{config["alg"]["MSE_LOSS_COEF"]}_vdn_' + '|'.join([str(x) for x in config["alg"]['NUM_BATCH']]) + '_' + '|'.join([str(x) for x in config["alg"]['NUM_UNILATERAL_BATCH']] if config["alg"].get("USE_UNILATERAL", False) else [])
    wandb.init(
        entity=config["ENTITY"],
        project=str(config["PROJECT_PREFIX"]) + 'overcooked_vdn',
        tags=[
            alg_name.upper(),
            env_name.upper(),
            "RNN",
            "TD_LOSS" if config["alg"].get("TD_LAMBDA_LOSS", True) else "DQN_LOSS",
            f"jax_{jax.__version__}",
        ],
        name=wandb_name,
        config=config,
        mode=config["WANDB_MODE"],
    )
    
    rng = jax.random.PRNGKey(config["SEED"])
    train_jit = jax.jit(make_train(config["alg"], env))
    with jax.disable_jit(config["DISABLE_JIT"]):
        outs = train_jit(rng)
    # rngs = jax.random.split(rng, config["NUM_SEEDS"])
    # train_vjit = jax.jit(jax.vmap(make_train(config["alg"], env)))
    # outs = jax.block_until_ready(train_vjit(rngs))


if __name__ == "__main__":
    main()
    
