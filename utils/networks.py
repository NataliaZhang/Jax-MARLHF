
from flax import linen as nn
import functools
from flax.linen.initializers import constant, orthogonal
import jax
import jax.numpy as jnp
import numpy as np
import distrax
from typing import Sequence, Dict
class ScannedRNN(nn.Module):
    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        """Applies the module."""
        rnn_state = carry
        ins, resets = x
        rnn_state = jnp.where(
            resets[:, np.newaxis],
            self.initialize_carry(carry.shape[0], carry.shape[1]),
            rnn_state,
        )
        new_rnn_state, y = nn.GRUCell(features=ins.shape[1])(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        # Use a dummy key since the default state init fn is just zeros.
        cell = nn.GRUCell(features=hidden_size)
        return cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, hidden_size))

class ActorRNN(nn.Module):
    action_dim: Sequence[int]
    config: Dict

    @nn.compact
    def __call__(self, hidden, x):
        if len(x) == 3:
            obs, dones, avail_actions = x
        else:
            obs, dones = x
            avail_actions = jnp.ones((obs.shape[0], obs.shape[1], self.action_dim))
        embedding = nn.Dense(
            128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(obs)
        embedding = nn.relu(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        actor_mean = nn.Dense(128, kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        unavail_actions = 1 - avail_actions
        action_logits = actor_mean - (unavail_actions * 1e10)

        pi = distrax.Categorical(logits=action_logits)

        return hidden, pi




class CriticRNN(nn.Module):
    
    @nn.compact
    def __call__(self, hidden, x):
        world_state, dones = x
        embedding = nn.Dense(
            hidden.shape[1], kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(world_state)
        embedding = nn.relu(embedding)
        
        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)
        
        critic = nn.Dense(128, kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )
        critic = nn.relu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )
        
        return hidden, jnp.squeeze(critic, axis=-1)
    
class ObsPretrainModel(nn.Module):
    action_dim: int
    hidden_size: int = 256
    action_embedding_size: int = 64
    obs_embedding_size: int = 64

    @nn.compact
    def __call__(self, hidden, trajectories):
        action = trajectories.action
        obs = trajectories.obs  # (max_seq_len, batch_size, obs_dim)
        dones = trajectories.done
        world_state = trajectories.world_state

        # change action into one-hot encoding
        action = jax.nn.one_hot(action, self.action_dim)

        embedded_action = nn.Dense(
            self.action_embedding_size, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(action)
        embedded_obs = nn.Dense(
            self.obs_embedding_size, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(obs)

        x = jnp.concatenate([embedded_action, embedded_obs, world_state], axis=-1)

        embedding = nn.Dense(
            self.hidden_size, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        embedding = nn.relu(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        prediction_layer = nn.Dense(self.hidden_size, kernel_init=orthogonal(2), bias_init=constant(0.0))(embedding)
        prediction_layer = nn.relu(prediction_layer)

        prediction = nn.Dense(obs.shape[-1], kernel_init=orthogonal(1.0), bias_init=constant(0.0))(prediction_layer)
        # prediction = jnp.squeeze(prediction, axis=-1)
        return hidden, prediction
 

class RewardModel(nn.Module):
    action_dim: int
    hidden_size: int = 256
    action_embedding_size: int = 64
    obs_embedding_size: int = 64

    @nn.compact
    def __call__(self, hidden, trajectories):
        action = trajectories.action
        obs = trajectories.obs  # (max_seq_len, batch_size, obs_dim)
        dones = trajectories.done
        world_state = trajectories.world_state

        # change action into one-hot encoding
        action = jax.nn.one_hot(action, self.action_dim)

        embedded_action = nn.Dense(
            self.action_embedding_size, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(action)
        embedded_obs = nn.Dense(
            self.obs_embedding_size, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(obs)

        x = jnp.concatenate([embedded_action, embedded_obs, world_state], axis=-1)

        embedding = nn.Dense(
            self.hidden_size, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        embedding = nn.relu(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        prediction_layer = nn.Dense(self.hidden_size, kernel_init=orthogonal(2), bias_init=constant(0.0))(embedding)
        prediction_layer = nn.relu(prediction_layer)

        prediction = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(prediction_layer)
        prediction = jnp.squeeze(prediction, axis=-1)
        return hidden, prediction
    
class RewardModelFF(nn.Module):
    action_dim: int
    layer_dim: int = 128 # 64
    action_embedding_size: int = 64
    obs_embedding_size: int = 64
    
    @nn.compact
    def __call__(self, unused_hidden, trajectories):
        action = trajectories.action
        obs = trajectories.obs  # (max_seq_len, batch_size, obs_dim)

        action = nn.one_hot(action, self.action_dim)  # (max_seq_len, batch_size, action_dim)

        # # one layer embedding for action and obs
        # x = jnp.concatenate([action, obs], axis=-1)
        # embedding = nn.Dense(
        #     features=self.layer_dim,
        #     kernel_init=orthogonal(np.sqrt(2)),
        #     bias_init=constant(0.0)
        # )(x)
        # prediction = nn.Dense(
        #     features=1,
        #     kernel_init=orthogonal(1.0),
        #     bias_init=constant(0.0)
        # )(embedding)

        embedded_action = nn.Dense(
            features=self.action_embedding_size,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0)
        )(action)
        embedded_action = nn.relu(embedded_action)
        embedded_action = nn.Dense(
            features=self.action_embedding_size,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0)
        )(action)
        embedded_action = nn.relu(embedded_action)

        embedded_obs = nn.Dense(
            features=self.obs_embedding_size,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0)
        )(obs)
        embedded_obs = nn.relu(embedded_obs)
        embedded_obs = nn.Dense(
            features=self.obs_embedding_size,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0)
        )(obs)
        embedded_obs = nn.relu(embedded_obs)

        x = jnp.concatenate([embedded_action, embedded_obs], axis=-1)

        embedding = nn.Dense(
            features=self.layer_dim,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0)
        )(x)
        embedding = nn.Dense(
            features=self.layer_dim,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0)
        )(x)    # deeper layer added
        embedding = nn.relu(embedding)
        prediction = nn.Dense(
            features=1,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0)
        )(embedding)
        
        prediction = jnp.squeeze(prediction, axis=-1)
        return unused_hidden, prediction
 
class AgentRNN(nn.Module):
    # homogenous agent for parameters sharing, assumes all agents have same obs and action dim
    action_dim: int
    hidden_dim: int
    init_scale: float
    config: dict = None
    @nn.compact
    def __call__(self, hidden, x):
        obs, dones = x
        embedding = nn.Dense(self.hidden_dim, kernel_init=orthogonal(self.init_scale), bias_init=constant(0.0))(obs)
        embedding = nn.relu(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)
        
        q_vals = nn.Dense(self.action_dim, kernel_init=orthogonal(self.init_scale), bias_init=constant(0.0))(embedding)

        return hidden, q_vals

class AgentFF(nn.Module):
    # homogenous agent for parameters sharing, assumes all agents have same obs and action dim
    action_dim: int
    hidden_dim: int
    init_scale: float
    config: dict = None
    @nn.compact
    def __call__(self, unused_hidden, x):
        obs, dones = x
        embedding = nn.Dense(self.hidden_dim, kernel_init=orthogonal(self.init_scale), bias_init=constant(0.0))(obs)
        embedding = nn.relu(embedding)
        embedding = nn.Dense(self.hidden_dim, kernel_init=orthogonal(self.init_scale), bias_init=constant(0.0))(embedding)
        q_vals = nn.Dense(self.action_dim, kernel_init=orthogonal(self.init_scale), bias_init=constant(0.0))(embedding)
        return unused_hidden, q_vals

def batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors, -1))

def timestep_batchify(x: dict, agent_list, num_actors=None):
    x = jnp.concatenate([x[a] for a in agent_list], axis=1)
    return x

def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_agents):
    x = x.reshape([num_agents, num_envs] + list(x.shape[1:]))
    return {a: x[i] for i, a in enumerate(agent_list)}

def timestep_unbatchify(x: jnp.ndarray, agent_list, num_envs, num_agents):
    x = x.reshape((x.shape[0], num_agents, num_envs, -1)).squeeze()
    return {a: x[:, i] for i, a in enumerate(agent_list)}

def print_jnp_shapes(d, key_path=None):
    if key_path is None:
        key_path = []
    for key, value in d.items():
        current_path = key_path + [key]
        if isinstance(value, dict):
            print_jnp_shapes(value, current_path)
        elif isinstance(value, jnp.ndarray):
            print("Key path:", " -> ".join(current_path), "Shape:", value.shape)
        else:
            print("Key path:", " -> ".join(current_path), "Shape:", value.shape)