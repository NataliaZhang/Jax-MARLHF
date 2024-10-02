# export PYTHONPATH="/homes/gws/nataz/JaxMARL"

import os

import jax.experimental
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax import struct
from flax import serialization
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any, Tuple, Union, Dict
import chex
import copy

from flax.training.train_state import TrainState
import distrax
import hydra
from omegaconf import DictConfig, OmegaConf
from functools import partial
import jaxmarl
from jaxmarl.wrappers.baselines import MPELogWrapper, JaxMARLWrapper
from jaxmarl.environments.multi_agent_env import MultiAgentEnv, State

import wandb
import functools
import matplotlib.pyplot as plt
import pickle
from utils.jax_dataloader import Trajectory
from utils.networks import ScannedRNN, ActorRNN, CriticRNN, RewardModel, RewardModelFF

    
class CooperativeEnvWrapper(JaxMARLWrapper):

    def __init__(self, env, agent_params, agent_model, hidden_size=128):
        super().__init__(env)
        self.model = agent_model
        self.params = agent_params
        self.agent_hidden_size = hidden_size
        self.max_obs_size = max([self._env.observation_space(agent).shape[-1] for agent in self._env.agents])
        self.agents = copy.copy(self._env.agents)
        self.agents.remove("agent_0")
        self.num_agents = len(self.agents)
    
    @partial(jax.jit, static_argnums=0)
    def reset(self,
              key):
        obs, env_state = self._env.reset(key)
        print("max_obs_size", self.max_obs_size)
        print("obs shape", {k: v.shape for k, v in obs.items()})
        obs = jax.tree.map(lambda x: jnp.pad(x, ((0, self.max_obs_size - x.shape[-1]))), obs)
        last_agent_obs = obs["agent_0"]
        last_agent_dones = jnp.zeros((1,), dtype=jnp.bool)
        agent_hidden_state = ScannedRNN.initialize_carry(1, self.agent_hidden_size).squeeze()
        del(obs["agent_0"])
        # del(env_state.env_state["agent_0"])

        print("last_obs shape", last_agent_obs.shape)
        print("obs dict shape", {k: v.shape for k, v in obs.items()})
        
        return obs, (env_state, agent_hidden_state, last_agent_obs, last_agent_dones)
    
    @partial(jax.jit, static_argnums=0)
    def step(self,
             key,
             state,
             action):
        env_state, agent_hidden_state, last_agent_obs, last_agent_dones = state
        last_agent_obs = last_agent_obs.reshape((1, 1, -1))
        last_agent_dones = last_agent_dones.reshape((1, -1))
        agent_hidden_state = agent_hidden_state.reshape((1, -1))
        agent_hidden_state, pi = self.model.apply(self.params, agent_hidden_state, (last_agent_obs, last_agent_dones))
        key_agent, key = jax.random.split(key)
        agent_action = pi.sample(seed=key_agent).squeeze()
        action["agent_0"] = agent_action
        obs, env_state, reward, done, info = self._env.step(
            key, env_state, action
        )
        obs = jax.tree.map(lambda x: jnp.pad(x, ((0, self.max_obs_size - x.shape[-1]))), obs)
        last_agent_obs = obs["agent_0"]
        last_agent_dones = done["agent_0"]
        print("last_agent_obs", last_agent_obs.shape)
        print("last_agent_dones", last_agent_dones.shape)
        last_agent_dones = jnp.expand_dims(last_agent_dones, axis=-1)

        del(obs["agent_0"])
        # del(env_state.env_state["agent_0"])
        del(reward["agent_0"])
        del(done["agent_0"])
        print("env.step")
        print("last_agent_obs", last_agent_obs.shape)
        print("agent_hidden_state", agent_hidden_state.shape)
        return obs, (env_state, agent_hidden_state.squeeze(), last_agent_obs, last_agent_dones), reward, done, info
   

class CooperativeEnvWrapperLoaded(CooperativeEnvWrapper):

    def __init__(self, env):
        good_model = ActorRNN(action_dim=env.action_space(env.agents[0]).n, config={})
        good_init_x = (
            jnp.zeros((1, 1, env.observation_space(env.agents[0]).shape[-1])),
            jnp.zeros((1, 1))
        )
        rng = jax.random.PRNGKey(0)
        good_init_hstate = ScannedRNN.initialize_carry(1, 128)
        good_network_params = good_model.init(rng, good_init_hstate, good_init_x)
        with open('model/tag/fixed_good_agent.msgpack', "rb") as f:
            good_params = serialization.from_bytes(good_network_params, f.read())
        super().__init__(env, good_params, good_model, hidden_size=128)