import os
import hydra
import time
import jax
import jax.numpy as jnp
import numpy as np
import jaxmarl
from jaxmarl.wrappers.baselines import JaxMARLWrapper
from jaxmarl.environments.mpe.simple import State
from jaxmarl.environments.mpe import MPEVisualizer
from flax import serialization
from flax.training.train_state import TrainState
import optax
from omegaconf import OmegaConf
from typing import NamedTuple
from functools import partial
import matplotlib.animation as animation
from utils.jax_dataloader import Trajectory
from utils.networks import ActorRNN, AgentRNN, ScannedRNN, RewardModel, RewardModelFF, batchify, unbatchify, timestep_batchify, print_jnp_shapes
from utils.tag_2_cooperative_wrapper import CooperativeEnvWrapperLoaded
import wandb
import pickle

class IL_Transition(NamedTuple):
    obs: jnp.ndarray
    done: jnp.ndarray
    teacher_act: jnp.ndarray
    info: jnp.ndarray

class vdn_env_wrapper(JaxMARLWrapper):
    """
    Add one-hot encoding of the agent id to the observation space.
    """
    def __init__(self, env):
        super().__init__(env)
        self.agents_one_hot = {a:oh for a, oh in zip(self.agents, jnp.eye(len(self.agents)))}
        self.max_action_space = env.action_space(env.agents[0]).n
    @partial(jax.jit, static_argnums=0)
    def step(self, rng, state, actions):
        obs_, state, reward, done, infos = self._env.step(rng, state, actions)
        obs = jax.tree_util.tree_map(self._preprocess_obs, {agent:obs_[agent] for agent in self.agents}, self.agents_one_hot)
        obs['__all__'] = jnp.concatenate([obs_[agent] for agent in self.agents], axis=-1)
        return obs, state, reward, done, infos
    @partial(jax.jit, static_argnums=0)
    def reset(self, rng):
        obs_, state = self._env.reset(rng)
        obs = jax.tree_util.tree_map(self._preprocess_obs, {agent:obs_[agent] for agent in self.agents}, self.agents_one_hot)
        obs['__all__'] = jnp.concatenate([obs_[agent] for agent in self.agents], axis=-1)
        return obs, state
    @partial(jax.jit, static_argnums=0)
    def _preprocess_obs(self, arr, extra_features):
        # flatten
        arr = arr.flatten()
        # concatenate the extra features
        arr = jnp.concatenate((arr, extra_features), axis=-1)
        return arr
    
def make_train(config):
    
    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    # if config["TEACHER_NETWORK_TYPE"] == "Qfn":
    if config['ENV_NAME'] == "MPE_simple_tag_v3":
        env = CooperativeEnvWrapperLoaded(env)
    env = vdn_env_wrapper(env)
    config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
    config["NUM_TEST_ACTORS"] = env.num_agents * config["NUM_TEST_ENVS"]
    
    student_model = ActorRNN(action_dim=env.action_space(env.agents[0]).n, config=config)
    
    offline_data_files = []
    config["OFFLINE_DATA_FILE"] = offline_data_files
    if config["ENV_NAME"] == "MPE_simple_spread_v3" and env.num_agents != 3:
        offline_data_name = config["ENV_NAME"] + "_na" + str(env.num_agents)
    else:
        offline_data_name = config["ENV_NAME"]

    if config["FILTER_BY_REWARD_MODEL"]:
        assert config["FILTER_RATIO"] <= 1.0 and config["FILTER_RATIO"] > 0.0, "FILTER_RATIO should be in (0, 1)"
        for offline_data_path, num_batchs in zip(config["OFFLINE_DATA_PATHS"][offline_data_name], config["NUM_BATCH"]):
            offline_data_files.extend([offline_data_path + f"traj_batch_{i}.pkl" for i in range(int(num_batchs / config["FILTER_RATIO"]))])
        if config["USE_UNILATERAL"]:
            for offline_data_path, num_batchs in zip(config["OFFLINE_UNILATERAL_DATA_PATHS"][offline_data_name], config["NUM_UNILATERAL_BATCH"]):
                offline_data_files.extend([offline_data_path + f"traj_batch_{i}.pkl" for i in range(int(num_batchs / config["FILTER_RATIO"]))])
    else:
        for offline_data_path, num_batchs in zip(config["OFFLINE_DATA_PATHS"][offline_data_name], config["NUM_BATCH"]):
            offline_data_files.extend([offline_data_path + f"traj_batch_{i}.pkl" for i in range(num_batchs)])
        if config["USE_UNILATERAL"]:
            for offline_data_path, num_batchs in zip(config["OFFLINE_UNILATERAL_DATA_PATHS"][offline_data_name], config["NUM_UNILATERAL_BATCH"]):
                offline_data_files.extend([offline_data_path + f"traj_batch_{i}.pkl" for i in range(num_batchs)])
        
    print("data ratio:", config["NUM_BATCH"])
    print("data ratio unilateral:", config["NUM_UNILATERAL_BATCH"])
    
    # load dataset
    dataset_obs = None
    dataset_done = None
    dataset_teacher_act = None
    def _load_data(idx):
        filename = config["OFFLINE_DATA_FILE"][int(idx)]
        print(f"Loading {filename}")
        with open(filename, 'rb') as f:
            traj_batch = pickle.load(f)
        traj_batch = traj_batch._replace(infos=None)
        return traj_batch
    
    if config["FILTER_BY_REWARD_MODEL"]:
        # load reward model
        if config["REWARD_MODEL_TYPE"] == "FF":
            reward_model = RewardModelFF(action_dim=env.max_action_space, layer_dim=config["FF_LAYER_DIM"])
            print("Using FF reward model, max action space:", env.max_action_space)
        elif config["REWARD_MODEL_TYPE"] == "RNN":
            config["RM_HIDDEN_SIZE"] = config["FF_LAYER_DIM"]
            reward_model = RewardModel(action_dim=env.max_action_space, hidden_size=config["RM_HIDDEN_SIZE"])
        rm_init_hstate = ScannedRNN.initialize_carry(1, config["RM_HIDDEN_SIZE"]) 
        _rm_init_rng = jax.random.PRNGKey(config["SEED"])
        dummy_traj = Trajectory(
            obs=jnp.zeros((1, 1, env.observation_space(env.agents[0]).shape[0])),
            action=jnp.ones((1, 1)),
            world_state=jnp.zeros((1, 1, env.observation_space(env.agents[0]).shape[0] * env.num_agents)),
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
        rm_model_path = os.path.join(config["REWARD_MODEL_PARAM_PATH"], "reward_model_best.msgpack")
        rm_params = jax.experimental.io_callback(_load_rm_model, rm_params, rm_params)

        # compute the predicted reward for each trajectory
        def _compute_reward(traj_batch):
            obs_batch = timestep_batchify(traj_batch.obs, env.agents)[..., :-env.num_agents]
            done_batch = timestep_batchify(traj_batch.dones, env.agents)
            action_batch = timestep_batchify(traj_batch.actions, env.agents)
            world_state = traj_batch.obs['__all__'].repeat(env.num_agents, axis=1)

            traj_to_predict = Trajectory(
                obs=obs_batch,  # removed the one-hot encoding of agents
                action=action_batch,
                world_state=world_state,
                done=done_batch,
            )
            _, reward = reward_model.apply(rm_params, ScannedRNN.initialize_carry(obs_batch.shape[1], config["RM_HIDDEN_SIZE"]), traj_to_predict)
            return reward
        print("traj_batch length:", len(config["OFFLINE_DATA_FILE"]))   # 10
        all_trajectories = [trajectory for trajectory in map(_load_data, range(len(config["OFFLINE_DATA_FILE"])))]
        print("all_trajectories length:", len(all_trajectories))   # 10
        rewards = map(_compute_reward, all_trajectories)

        rewards = jnp.stack(list(rewards), axis=2)    # (num_traj, num_steps, num_agents * num_envs)
        print(f"Shape of the concatenated rewards: {rewards.shape}")
        rewards = jnp.mean(rewards, axis=(0,1))
        print("rewards: ",  rewards)
        print("rewards.shape: ", rewards.shape)
    
        # filter out the trajectories with relatively low rewards such that filter_ratio of the rewards is above a threshold
        sorted_rewards = jnp.sort(rewards, descending=True)
        reward_threshold = sorted_rewards[int(len(sorted_rewards) * config["FILTER_RATIO"])]
        filtered_traj_batch = []
        for traj_batch, reward in zip(all_trajectories, rewards):
            if reward > reward_threshold:
                filtered_traj_batch.append(traj_batch)

        print(f"Filtered out {len(config['OFFLINE_DATA_FILE']) - len(filtered_traj_batch)} out of {len(config['OFFLINE_DATA_FILE'])} trajectories with rewards below {reward_threshold}")

    print("Loading dataset...")
    start_time = time.time()
    for idx in range(len(config["OFFLINE_DATA_FILE"])):
        def _original2IL(original_traj):
            obs = timestep_batchify(original_traj.obs, env.agents)[..., :-len(env.agents)] # remove the agents one-hot encoding
            teacher_act = timestep_batchify(original_traj.actions, env.agents)
            done = timestep_batchify(original_traj.dones, env.agents)
            return obs, done, teacher_act
        
        if config["FILTER_BY_REWARD_MODEL"]:
            if idx >= len(filtered_traj_batch):
                break
            filtered_traj_batch_ = filtered_traj_batch[idx]
            original_traj_batch = _load_data(idx)
            assert original_traj_batch.obs['__all__'].shape == _load_data(idx).obs['__all__'].shape
            new_obs, new_done, new_teacher_act = _original2IL(filtered_traj_batch_)
        else:
            original_traj_batch = _load_data(idx)
            # make batchs for IL
            new_obs, new_done, new_teacher_act = _original2IL(original_traj_batch)
        dataset_obs = new_obs if dataset_obs is None else jnp.concatenate((dataset_obs, new_obs), axis=1)
        dataset_done = new_done if dataset_done is None else jnp.concatenate((dataset_done, new_done), axis=1)
        dataset_teacher_act = new_teacher_act if dataset_teacher_act is None else jnp.concatenate((dataset_teacher_act, new_teacher_act), axis=1)

    config["DATASET_SIZE"] = dataset_obs.shape[1]
    IL_dataset = IL_Transition(obs=dataset_obs, done=dataset_done, teacher_act=dataset_teacher_act, info=None)
    
    def train(rng):
        # initialize environment
        rng, _env_rng = jax.random.split(rng)
        _env_rngs = jax.random.split(_env_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(_env_rngs)
        
        # initialize learner model
        rng, _s_init_rng = jax.random.split(rng)
        _s_init_x = (
            jnp.zeros((1, 1, env.observation_space(env.agents[0]).shape[0])),
            jnp.zeros((1, 1))
        )
        _s_init_h = ScannedRNN.initialize_carry(1, config["STUDENT_NETWORK_HIDDEN"])
        s_network_params = student_model.init(_s_init_rng, _s_init_h, _s_init_x)
        
        
        # initialize train state and optimizer
        student_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=optax.linear_schedule(init_value=config["LR"], end_value=1e-4, transition_steps=config["NUM_EPOCHS"] * config["DATASET_SIZE"] / config["BATCH_SIZE"]), eps=1e-5),
            )
        student_train_state = TrainState.create(
            apply_fn=student_model.apply,
            params=s_network_params,
            tx=student_tx,
        )
        
        def _train_epochs_and_one_test(runner_state, tested_times):
            student_train_state, rng, best_test_return = runner_state
            def _train_epoch(IL_training_state, after_test):
                student_train_state, init_s_h_state, IL_traj_batch, rng = IL_training_state
                def _train_minibatch(student_train_state, minibatch):
                    s_h_state, IL_traj = minibatch
                    def _IL_loss_fn(student_params, s_h_state, IL_traj):
                        obs = IL_traj.obs
                        done = IL_traj.done
                        teacher_act = IL_traj.teacher_act
                        
                        # forward pass
                        in_data = (obs, done)
                        s_h_state, pi = student_model.apply(student_params, s_h_state.squeeze(), in_data)
                        
                        
                        # compute loss
                        loss = -pi.log_prob(teacher_act)
                        loss = jnp.mean(loss)
                        return loss
                    grad_fn = jax.value_and_grad(_IL_loss_fn)
                    loss, grad = grad_fn(student_train_state.params, s_h_state, IL_traj)
                    student_train_state = student_train_state.apply_gradients(grads=grad)
                    return student_train_state, loss
                # prepare minibatches
                batch = (
                    init_s_h_state,
                    IL_traj_batch,
                )
                # shuffle and separate into minibatches
                per_rng, rng = jax.random.split(rng)
                permutation = jax.random.permutation(per_rng, config["DATASET_SIZE"])
                shuffled_batch = jax.tree.map(lambda x: jnp.take(x, permutation, axis=1), batch)
                minibatches = jax.tree.map(
                    lambda x: jnp.swapaxes(jnp.reshape(x, [x.shape[0], -1, config["BATCH_SIZE"]] + list(x.shape[2:])), 1, 0),
                    shuffled_batch,
                )
                student_train_state, IL_losses = jax.lax.scan(
                    _train_minibatch, student_train_state, minibatches
                )
                IL_training_state = (
                    student_train_state,
                    init_s_h_state,
                    IL_traj_batch,
                    rng,
                )
                return IL_training_state, IL_losses

            IL_training_state = (
                student_train_state,
                jnp.expand_dims(ScannedRNN.initialize_carry(config["DATASET_SIZE"], config["STUDENT_NETWORK_HIDDEN"]), 0),
                IL_dataset,
                rng,
            )
            
            IL_training_state, IL_losses = jax.lax.scan(
                _train_epoch, IL_training_state, jnp.arange(config["TEST_INTERVAL"])
            )
            IL_loss = IL_losses.mean()
            
            student_train_state, new_s_h_state, _, rng = IL_training_state
            new_s_h_state = new_s_h_state.squeeze()
            # test the student model
            def _test_step(test_state, unused):
                s_state, test_env_state, test_obs, test_done, test_h_state, rng = test_state
                test_obs_batch = batchify(test_obs, env.agents, config["NUM_TEST_ACTORS"])
                if config["TEACHER_NETWORK_TYPE"] == "Qfn":
                    test_obs_batch = test_obs_batch[..., :-env.num_agents] # remove the one-hot encoding of the agent id, which is not needed for StudentRNN
                test_in = (test_obs_batch[np.newaxis, :], test_done[np.newaxis, :])
                
                test_h_state, pi = student_model.apply(s_state.params, test_h_state, test_in)
                test_act = pi.sample(seed=rng)[0]
                
                test_act = unbatchify(test_act, env.agents, config["NUM_TEST_ENVS"], env.num_agents)
                _test_rng, rng = jax.random.split(rng)
                _test_rngs = jax.random.split(_test_rng, config["NUM_TEST_ENVS"])
                test_obs, test_env_state, test_rewards, test_dones, test_info = jax.vmap(env.step, in_axes=(0, 0, 0))(
                    _test_rngs, test_env_state, test_act)
                test_info = jax.tree_util.tree_map(lambda x: x.reshape((config["NUM_TEST_ACTORS"])), test_info)
                test_done_batch = batchify(test_dones, env.agents, config["NUM_TEST_ACTORS"]).squeeze()
                test_state = (s_state, test_env_state, test_obs, test_done_batch, test_h_state, rng)
                
                return test_state, (test_rewards, test_env_state, test_obs, test_dones)
            
            _test_rng, rng = jax.random.split(rng)
            _test_rngs = jax.random.split(_test_rng, config["NUM_TEST_ENVS"])
            test_obsv, test_env_state = jax.vmap(env.reset, in_axes=(0,))(_test_rngs)
            test_state = (
                student_train_state,
                test_env_state,
                test_obsv,
                jnp.zeros((config["NUM_TEST_ACTORS"]), dtype=bool), # test_dones
                ScannedRNN.initialize_carry(config["NUM_TEST_ACTORS"], config["STUDENT_NETWORK_HIDDEN"]),
                rng,
            )
            test_state, test_rewards_states=jax.lax.scan(
                _test_step, test_state, jnp.arange(config["NUM_TEST_STEPS"])
            )
            test_rewards, test_states, test_obs, test_dones = test_rewards_states
            student_train_state = test_state[0]

            # compute the metrics of the first episode that is done for each parallel env
            def first_episode_returns(rewards, dones):
                first_done = jax.lax.select(jnp.argmax(dones)==0., dones.size, jnp.argmax(dones))
                first_episode_mask = jnp.where(jnp.arange(dones.size) <= first_done, True, False)
                return jnp.where(first_episode_mask, rewards, 0.).sum()
            
            print("test_rewards.shape as a dict", {k: v.shape for k, v in test_rewards.items()})
            test_rewards['__all__'] = jnp.array([v for k, v in test_rewards.items()]).sum(axis=0)
            print("test_rewards.shape", test_rewards['__all__'].shape)
            print("test_dones.shape", test_dones['__all__'].shape)

            first_test_returns = jax.tree.map(lambda r: jax.vmap(first_episode_returns, in_axes=1)(r, test_dones['__all__']), test_rewards)
            print("first_test_returns.shape", {k: v.shape for k, v in first_test_returns.items()})
            test_return = first_test_returns['__all__'].mean() / env.num_agents

            def callback(params, tested_times, test_return, best_test_return, IL_loss, test_states, test_obs):
                epoch = tested_times * config["TEST_INTERVAL"]
                wandb.log({"test_return": test_return,
                           "IL_loss": IL_loss,},
                          step=epoch)
                print(f"Epoch: {epoch}, test return: {test_return}, IL loss: {IL_loss}")

                if tested_times == config["NUM_EPOCHS"]//config["TEST_INTERVAL"] - 1:
                    if not os.path.exists(config["STUDENT_NETWORK_SAVE_PATH"]):
                        os.makedirs(config["STUDENT_NETWORK_SAVE_PATH"])
                    file_path = os.path.join(config["STUDENT_NETWORK_SAVE_PATH"], "|".join([str(r) for r in config["NUM_BATCH"]]) + 'final.msgpack')
                    with open(file_path, "wb") as f:
                        f.write(serialization.to_bytes(params))
                    print(f"Saved the best model to {file_path} with test return {test_return}")
                    if config["ENV_NAME"] == "MPE_simple_tag_v3":
                        state_seq_in_one_State = test_states[0]
                    else:
                        state_seq_in_one_State = test_states
                    state_seq = []
                    for i in range(config["NUM_TEST_STEPS"]):
                        state_seq.append(State(
                            p_pos=state_seq_in_one_State.p_pos[i, 0],
                            p_vel=state_seq_in_one_State.p_vel[i, 0],
                            c=state_seq_in_one_State.c[i, 0],
                            done=state_seq_in_one_State.done[i, 0],
                            step=state_seq_in_one_State.step[i, 0],
                            goal=None,
                            ))
                    viz = MPEVisualizer(env, state_seq)
                    ani = animation.FuncAnimation(
                        viz.fig,
                        viz.update,
                        frames=len(viz.state_seq),
                        blit=False,
                        interval=viz.interval,
                    )
                    file_path = config["VIS_SAVE_PATH"]
                    if not os.path.exists(file_path):
                        os.makedirs(file_path)
                    vis_save_path = os.path.join(file_path, '|'.join([str(r) for r in config["NUM_BATCH"]]) + f'vdn_student_{test_return}.gif')
                    ani.save(vis_save_path, writer='imagemagick', fps=15)
                    print("Visualized gif saved at ", vis_save_path)
            jax.experimental.io_callback(callback, None, student_train_state.params, tested_times, test_return, best_test_return, IL_loss, test_states, test_obs)
            
            runner_state = (
                student_train_state,
                rng,
                jnp.maximum(best_test_return, test_return),
            )
            
            return runner_state, (test_return, IL_loss)
        
        runner_state = (
            student_train_state,
            rng,
            float('-inf') # best return
        )
        
        runner_state, metric = jax.lax.scan(
            _train_epochs_and_one_test, runner_state, np.arange(config["NUM_EPOCHS"]//config["TEST_INTERVAL"])
        )
        print(f"Training finished after {config['NUM_EPOCHS']} epochs.")
        return {"runner_state": runner_state, "metric": metric}
    
    return train

@hydra.main(version_base=None, config_path="config/", config_name="offline_IL_config")
def main(config):
    config = OmegaConf.to_container(config)
    if config["ENV_NAME"].split("_")[0] == 'MPE':
        env_name_str = config["ENV_NAME"].split("_")[2]
    else:
        env_name_str = config["ENV_NAME"]

    if env_name_str == "spread":
        if config["USE_UNILATERAL"]:
            config["REWARD_MODEL_PARAM_PATH"] = f'results/{env_name_str}/na{config["ENV_KWARGS"]["num_agents"]}/unilateral/rm_{config["REWARD_MODEL_TYPE"]}_{config["FF_LAYER_DIM"]}_mse_{config["MSE_LOSS_COEF"]}_vdn_' + '|'.join([str(x) for x in config['NUM_BATCH']]) + "_" + '|'.join([str(x) for x in config['NUM_UNILATERAL_BATCH']])
        else:
            config["REWARD_MODEL_PARAM_PATH"] = f'results/{env_name_str}/na{config["ENV_KWARGS"]["num_agents"]}/rm_{config["REWARD_MODEL_TYPE"]}_{config["FF_LAYER_DIM"]}_mse_{config["MSE_LOSS_COEF"]}_vdn_' + '|'.join([str(x) for x in config['NUM_BATCH']])
    else:
        if config["USE_UNILATERAL"]:
            config["REWARD_MODEL_PARAM_PATH"] = f'results/{env_name_str}/unilateral/rm_{config["REWARD_MODEL_TYPE"]}_{config["FF_LAYER_DIM"]}_mse_{config["MSE_LOSS_COEF"]}_vdn_' + '|'.join([str(x) for x in config['NUM_BATCH']]) + "_" + '|'.join([str(x) for x in config['NUM_UNILATERAL_BATCH']])
        else:
            config["REWARD_MODEL_PARAM_PATH"] = f'results/{env_name_str}/rm_{config["REWARD_MODEL_TYPE"]}_{config["FF_LAYER_DIM"]}_mse_{config["MSE_LOSS_COEF"]}_vdn_' + '|'.join([str(x) for x in config['NUM_BATCH']])

    wandb_project = str(config["PROJECT_PREFIX"]) + env_name_str + "_IL"

    if config["DEBUG"]:
        config["WANDB_MODE"] = "disabled"
        config["TOTAL_TIMESTEPS"] = 1e5
    ratio_name = "|".join([str(r) for r in config["NUM_BATCH"]]) + '_' + "|".join([str(r) for r in config["NUM_UNILATERAL_BATCH"]])
    wandb_name = "offline_IL_" + ratio_name
    wandb.init(
        entity=config["ENTITY"],
        project=wandb_project,
        tags=["None"],
        name=wandb_name,
        config=config,
        mode=config["WANDB_MODE"],
        # mode='disabled',
    )
    
    train = make_train(config)
    with jax.disable_jit(config["DISABLE_JIT"]):
        jit_train = jax.jit(train)
        rng = jax.random.PRNGKey(config["SEED"])
        output = jit_train(rng)
    wandb.finish()
        
if __name__ == "__main__":
    main()