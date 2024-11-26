import pickle
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '6'
import jax.numpy as jnp
import time
import jax.random
from typing import NamedTuple
from utils.networks import timestep_batchify, batchify

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

class Trajectory(NamedTuple):
    obs: jnp.ndarray
    action: jnp.ndarray
    world_state: jnp.ndarray = None
    done: jnp.ndarray = None
    reward: jnp.ndarray = None
    log_prob: jnp.ndarray = None
    avail_actions: jnp.ndarray = None


class JaxDataLoader:
    def __init__(self, dir_path, file_list, env, seed=0, debug=True, batch_size=128, need_reward=False, vdn=False, device='cpu'):
        self.dir_path = dir_path
        self.file_list = file_list
        self.env = env
        self.seed = seed
        self.max_length = None
        self.batch_size = batch_size
        self.debug = debug
        self.rng = jax.random.PRNGKey(seed)
        self.need_reward = need_reward
        self.batchs = None
        self.vdn = vdn
        self.device = device
        
        if debug:
            print("Loading data from ", dir_path)
            print("file_list: ", file_list)
            start = time.time()
        
        self.load_vdn_data()   
        
    def load_vdn_data(self):
        start_time = time.time()
        obs = None
        action = None
        done = None
        reward = None
        world_state = None
        agents = self.env.agents
        def convert_data(data):
            obs = data.obs 
            action = data.action
            done = data.done
            reward = data.reward
            world_state = obs.copy()
            return obs, action, done, reward, world_state
        
        if len(self.file_list) == 0:
            self.vdn_data = None
            raise ValueError("No files to load")
        
        for file in self.file_list:
            print("Loading file from ", os.path.join(self.dir_path, file))
            with open(os.path.join(self.dir_path, file), 'rb') as f:
                new_data = pickle.load(f)
                new_obs, new_action, new_done, new_reward, new_world_state = convert_data(new_data)
                obs = jnp.concatenate([obs, new_obs], axis=1) if obs is not None else new_obs
                action = jnp.concatenate([action, new_action], axis=1) if action is not None else new_action
                done = jnp.concatenate([done, new_done], axis=1) if done is not None else new_done
                reward = jnp.concatenate([reward, new_reward], axis=1) if reward is not None else new_reward
                world_state = jnp.concatenate([world_state, new_world_state], axis=1) if world_state is not None else new_world_state

        obs = obs.swapaxes(0, 1)        
        action = action.swapaxes(0, 1).squeeze()
        done = done.swapaxes(0, 1)
        # reward = reward.squeeze()
        reward = reward.swapaxes(0, 1)
        # print("reward shape: ", reward.shape)
        world_state = world_state.swapaxes(0, 1)
        traj_lengths = jnp.ones((len(obs),)) * 26    
        self.vdn_data = (jnp.array(obs), jnp.array(action), jnp.array(world_state), jnp.array(done), jnp.array(reward), traj_lengths)
        if self.debug:
            print("Vdn data from {} files loaded".format(len(self.file_list)))
            print("Data loaded in ", time.time() - start_time, " seconds")
            
    def __len__(self):
        if self.vdn_data is None:
            return 0
        if self.vdn:
            return self.vdn_data[0].shape[0]
        else:
            return len(self.data['trajs'])
    
    def get_data_for_jit(self):
        if self.vdn:
            obs, action, world_state, done, reward, traj_lengths= self.vdn_data
            return obs, action, world_state, done, None, None, reward.sum(axis=1), traj_lengths
        """convert all the data into a huge array for jit"""
        if self.debug:
            print("Converting data into a huge array for jit...")
            start_time = time.time()
        obs = []
        action = []
        world_state = []
        done = []
        reward = []
        log_prob = []
        avail_actions = []
        for traj in self.data['trajs'][:]:
            # filter out too-long trajs
            if traj.obs.shape[0] > 64:
                continue
            obs.append(traj.obs)
            action.append(traj.action)
            world_state.append(traj.world_state)
            done.append(traj.done)
            log_prob.append(traj.log_prob)
            if traj.avail_actions is not None:
                avail_actions.append(traj.avail_actions)
            if self.need_reward:
                reward.append(traj.reward)
        maximum_data_idx = len(obs)//self.batch_size * self.batch_size
        obs = obs[:maximum_data_idx]
        action = action[:maximum_data_idx]
        world_state = world_state[:maximum_data_idx]
        done = done[:maximum_data_idx]
        log_prob = log_prob[:maximum_data_idx]
        if self.need_reward:
            reward = reward[:maximum_data_idx]
        if len(avail_actions) > 0:
            avail_actions = avail_actions[:maximum_data_idx]
        

        def pad_and_concatenate(data, max_length=None):
            if max_length is None:
                max_length = max([d.shape[0] for d in data])
            padded_data = []
            if len(data[0].shape) == 1:
                for d in data:
                    padded_data.append(jnp.pad(d, (0, max_length - d.shape[0]), mode='constant', constant_values=0))
                return jnp.array(padded_data)
            else:
                for d in data:
                    padded_data.append(jnp.pad(d, ((0, max_length - d.shape[0]), (0, 0)), mode='constant', constant_values=0))
            return jnp.array(padded_data)
        
        obs = pad_and_concatenate(obs)
        action = pad_and_concatenate(action)
        world_state = pad_and_concatenate(world_state)
        done = pad_and_concatenate(done)
        log_prob = pad_and_concatenate(log_prob)
        returns = jnp.array(self.data['rewards'][:maximum_data_idx])
        if len(avail_actions) > 0:
            avail_actions = pad_and_concatenate(avail_actions)
        if self.need_reward:
            reward = pad_and_concatenate(reward)
        traj_lengths = jnp.array(self.data['traj_lens'][:maximum_data_idx])
        
        if self.debug:
            print("obs shape: ", obs.shape)
            print("action shape: ", action.shape)
            print("world_state shape: ", world_state.shape)
            print("done shape: ", done.shape)
            print("rewards shape: ", returns.shape)
            print("log_prob shape: ", log_prob.shape)
            print("traj_lengths shape: ", traj_lengths.shape)
            if self.need_reward:
                print("reward shape: ", reward.shape)
            else:
                print("reward shape: Not needed")
            if len(avail_actions) > 0:
                print("avail_actions shape: ", avail_actions.shape)
            else:
                print("avail_actions shape: None")
            print("Data converted in ", time.time() - start_time, " seconds")
        if self.need_reward:
            return obs, action, world_state, done, log_prob, avail_actions, returns, traj_lengths, reward
        else:
            return obs, action, world_state, done, log_prob, avail_actions, returns, traj_lengths
    
    
    def get_dummy_batch(self, size=1, need_avail_actions=False):
        # return a dummy batch of data with the same shape as the real batch
        if self.vdn:
            obs_dim = self.vdn_data[0].shape[-1]
            world_state_dim = self.vdn_data[2].shape[-1]
        else:
            obs_dim = self.data['trajs'][0].obs.shape[-1]
            world_state_dim = self.data['trajs'][0].world_state.shape[-1]
        if self.debug:
            print("obs_dim: ", obs_dim)
            print("world_state_dim: ", world_state_dim)
        dummy_obs = jnp.zeros((1, size, obs_dim))
        dummy_action = jnp.zeros((1, size)) # mpe agent action is a scalar
        dummy_world_state = jnp.zeros((1, size, world_state_dim))
        dummy_done = jnp.zeros((1, size))
        dummy_log_prob = jnp.zeros((1, size))
        dummy_reward = jnp.zeros((1, size))
        if need_avail_actions:
            avail_actions = jnp.zeros((1, size, self.data['trajs'][0].avail_actions.shape[-1]))
        else:
            avail_actions = None
        dummy_trajs = Trajectory(obs=dummy_obs, 
                                 action=dummy_action, 
                                 world_state=dummy_world_state,
                                 done=dummy_done,
                                 log_prob=dummy_log_prob,
                                 reward=dummy_reward,
                                 avail_actions=avail_actions)
        dummy_rewards = jnp.zeros((size,))
        
        return dummy_trajs, dummy_rewards, jnp.ones((size,))



def main():
    dir_path = "data/vdn104"
    file_list = ["traj_batch_" + str(x) + ".pkl" for x in range(10)]
    dataloader = JaxDataLoader(dir_path, file_list, vdn=True)
    obs, action, world_state, done, log_prob, avail_actions, rewards, traj_lens = dataloader.get_data_for_jit()
    
if __name__ == "__main__":
    main()