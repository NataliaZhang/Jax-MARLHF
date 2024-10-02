import os
import jax
import jax.experimental
import jax.numpy as jnp
import jaxmarl
import numpy as np
from jax import grad, jit, vmap
from flax import linen as nn
from flax.linen.initializers import orthogonal, constant
from flax.training.train_state import TrainState
from flax import serialization
import optax
import hydra
import functools
from omegaconf import OmegaConf
import wandb
from sklearn.model_selection import train_test_split

from utils.jax_dataloader import JaxDataLoader, Trajectory, Transition
from utils.networks import ScannedRNN, RewardModelFF, RewardModel, print_jnp_shapes, ObsPretrainModel
from utils.tag_2_cooperative_wrapper import CooperativeEnvWrapperLoaded

    
def save_model(params, model_dir='results/mpe_reward_model', model_name="reward_model_final.msgpack"):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    with open(os.path.join(model_dir, model_name), 'wb') as f:
        f.write(serialization.to_bytes(params))
    print(f"Model saved to {model_dir}/{model_name}")

def load_model(init_params, model_dir, model_name="pretrain_reward_model_best.msgpack"):
    with open(os.path.join(model_dir, model_name), 'rb') as f:
        params = serialization.from_bytes(init_params, f.read())
    print(f"Model loaded from {model_dir}/{model_name}")
    return params


def predict_reward(model, params, hidden, traj, seq_lens):
    def compute_masked_sum(predictions, seq_lens):
        range_matrix = jnp.arange(predictions.shape[0]) # (max_seq_len,)
        mask = range_matrix[:, None] < seq_lens  # (max_seq_len, batch_size)
        masked_predictions = jnp.where(mask, predictions, 0)
        return jnp.sum(masked_predictions, axis=0)  # (batch_size,)
        
    hidden, reward = model.apply(params, hidden, traj)
    squared_residual_reward = jnp.square(reward[1:] - reward[:-1])
    reward = compute_masked_sum(reward, seq_lens)
    squared_residual_reward = compute_masked_sum(squared_residual_reward, seq_lens-1)
    return hidden, reward, squared_residual_reward

def make_train_model(model, dataloader, config):

    def lr_schedule(count):
        iteration = 200
        frac = 0.5 * (1 + jnp.cos(jnp.pi * (count % iteration) / iteration))
        return config["LR"] * frac
    
    def train(rng):

        # INIT NETWORK
        def create_train_state(rng, trajs_0):
            init_hstate = ScannedRNN.initialize_carry(config["MINIBATCH_SIZE"], config["RNN_HIDDEN_SIZE"])

            params = model.init(rng, init_hstate, trajs_0)
            
            if config["LOAD_FROM_PRETRAIN"]:
                pretrain_model = ObsPretrainModel(hidden_size=config["RNN_HIDDEN_SIZE"], action_dim=model.action_dim)
                pretrain_init_params = pretrain_model.init(rng, init_hstate, trajs_0)
                pretrain_init_params = load_model(pretrain_init_params, model_dir=config["SAVE_REWARD_MODEL_PATH"], model_name=config["PRETRAIN_MODEL_NAME"])
                # keep all the params except the last layer
                pretrain_init_params['params']["Dense_4"] = params['params']["Dense_4"]
                params = pretrain_init_params
            
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adamw(learning_rate=lr_schedule, weight_decay=1e-4, eps=1e-5),  # add weight decay
            )
            return TrainState.create(apply_fn=model.apply, params=params, tx=tx)
        
        rng, _rng = jax.random.split(rng)
        train_ratio = 0.9
        config["REWARD_MODEL_MINIBATCHES"] = (int(len(dataloader) * train_ratio) // config["MINIBATCH_SIZE"])
        config["REWARD_MODEL_TEST_MINIBATCHES"] = int(len(dataloader) // config["MINIBATCH_SIZE"]) - config["REWARD_MODEL_MINIBATCHES"]
        print("Reward model minibatches: ", config["REWARD_MODEL_MINIBATCHES"])
        print("Reward model test minibatches: ", config["REWARD_MODEL_TEST_MINIBATCHES"])
        print("dataloder length: ", int(len(dataloader)))
        config["NUM_PAIRS"] = config["MINIBATCH_SIZE"] * 10

        dummy_trajs, _, _ = dataloader.get_dummy_batch(size=config["MINIBATCH_SIZE"])
        train_state = create_train_state(_rng, dummy_trajs)

        # INIT UPDATE STATE
        update_state = (
            train_state, 
            rng
        )
        whole_dataset = dataloader.get_data_for_jit()
        shuffle_idx = jax.random.permutation(rng, whole_dataset[0].shape[0])
        whole_dataset = jax.tree_util.tree_map(lambda x: jnp.take(x, shuffle_idx, axis=0), whole_dataset)
        
        dataset = []
        testset = []
        train_data_size = config["REWARD_MODEL_MINIBATCHES"] * config["MINIBATCH_SIZE"]
        test_data_size = config["REWARD_MODEL_TEST_MINIBATCHES"] * config["MINIBATCH_SIZE"]
        for item in whole_dataset:
            if item is not None:
                dataset.append(item[:train_data_size])
                testset.append(item[train_data_size:])   
            else:
                dataset.append(None)
                testset.append(None)

        # TRAINING LOOP
        def train_epoch(update_runner_state, unused):
            update_state, dataset, testset, update_epochs, best_test_loss = update_runner_state
            
            def train_minibatch(batch_state, minibatch):
                state, rng = batch_state
                minibatch = jax.device_put(minibatch, jax.devices("gpu")[0])
                init_hstate, trajs, rewards, seq_lens = minibatch

                def compute_loss(params, init_hstate, trajs, rewards, seq_lens, indices, rng):
                    _, predicted_rewards, squared_residual_reward = predict_reward(model, params, init_hstate.transpose(), trajs, seq_lens) # (batch_size,)

                    predicted_reward_diffs = predicted_rewards[indices[0]] - predicted_rewards[indices[1]]  # shape (num_pairs,)
                    reward_pairs = jnp.take(rewards[0], indices, axis=0)
                    reward_diffs = reward_pairs[0] - reward_pairs[1]  # shape (num_pairs,)
                    reward_diffs = reward_diffs / max_reward_diff   # (0, 1]
                    preference_probs = 1 / (1 + jnp.exp(-reward_diffs * config["STEEPNESS"]))
                    preferences = jax.random.bernoulli(rng, preference_probs, shape=(config["NUM_PAIRS"],)) * 2 - 1   # sequnce of {-1, 1} (num_pairs,)
                    
                    sigmoid = jax.nn.sigmoid(preferences * predicted_reward_diffs)  # (num_pairs,)
                    ground_truth_sigmoid = jax.nn.sigmoid(preferences * reward_diffs * config["STEEPNESS"])  # (num_pairs,)
                    
                    value_loss = - jnp.log(1e-6 + sigmoid)    # (num_pairs,)
                    value_loss = jnp.mean(value_loss)
                    baseline_loss = - jnp.log(1e-6 + ground_truth_sigmoid).mean()    # (num_pairs,)

                    if config["DEBUG"]:
                        print("sigmoid[:5]: ", sigmoid[:5])
                        print("ground_truth_sigmoid[:5]: ", ground_truth_sigmoid[:5])
                        raise ValueError("Stop")
                    
                    value_loss = value_loss - baseline_loss
                    
                    mse_loss = jnp.mean(squared_residual_reward)

                    total_loss = value_loss + config["MSE_LOSS_COEF"] * mse_loss

                    return total_loss, (value_loss, mse_loss, baseline_loss)
                
                grad_fn = jax.value_and_grad(compute_loss, has_aux=True)
                rng, id_rng, l_rng = jax.random.split(rng, 3)
                indices = jax.random.randint(id_rng, (2, config["NUM_PAIRS"]), 0, config["MINIBATCH_SIZE"])

                loss, grad = grad_fn(state.params, init_hstate, trajs, rewards, seq_lens, indices, l_rng)
                new_state = state.apply_gradients(grads=grad)
                loss_info = {
                    "total_loss": loss[0],
                    "value_loss": loss[1][0],
                    "mse_loss": loss[1][1],
                    "baseline_loss": loss[1][2],
                }

                new_batch_state = (new_state, rng)
                return new_batch_state, loss_info
            
            def test_minibatch(batch_state, minibatch):
                state, rng = batch_state
                minibatch = jax.device_put(minibatch, jax.devices("gpu")[0])
                init_hstate, trajs, rewards, seq_lens = minibatch
                
                def compute_loss(params, init_hstate, trajs, rewards, seq_lens, indices, rng):
                    _, predicted_rewards, squared_residual_reward = predict_reward(model, params, init_hstate.transpose(), trajs, seq_lens) # (batch_size,)

                    predicted_reward_diffs = predicted_rewards[indices[0]] - predicted_rewards[indices[1]]  # shape (num_pairs,)

                    reward_pairs = jnp.take(rewards[0], indices, axis=0)
                    reward_diffs = reward_pairs[0] - reward_pairs[1]  # shape (num_pairs,)
                    reward_diffs = reward_diffs / max_reward_diff   # (0, 1]
                    preference_probs = 1 / (1 + jnp.exp(-reward_diffs * config["STEEPNESS"]))
                    preferences = jax.random.bernoulli(rng, preference_probs, shape=(config["NUM_PAIRS"],)) * 2 - 1   # sequnce of {-1, 1} (num_pairs,)
                    
                    sigmoid = jax.nn.sigmoid(preferences * predicted_reward_diffs)  # (num_pairs,)
                    ground_truth_sigmoid = jax.nn.sigmoid(preferences * reward_diffs * config["STEEPNESS"])  # (num_pairs,)
                    
                    value_loss = - jnp.log(1e-6 + sigmoid)    # (num_pairs,)
                    value_loss = jnp.mean(value_loss)
                    baseline_loss = - jnp.log(1e-6 + ground_truth_sigmoid).mean()    # (num_pairs,)
                    
                    value_loss = value_loss - baseline_loss
                    
                    mse_loss = jnp.mean(squared_residual_reward)
                    
                    total_loss = value_loss + config["MSE_LOSS_COEF"] * mse_loss

                    return total_loss, (value_loss, mse_loss, baseline_loss)
                
                rng, id_rng, l_rng = jax.random.split(rng, 3)
                indices = jax.random.randint(id_rng, (2, config["NUM_PAIRS"]), 0, config["MINIBATCH_SIZE"])
                
                test_loss = compute_loss(state.params, init_hstate, trajs, rewards, seq_lens, indices, l_rng)
                test_loss_info = {
                    "total_loss": test_loss[0],
                    "value_loss": test_loss[1][0],
                    "mse_loss": test_loss[1][1],
                    "baseline_loss": test_loss[1][2],
                }
                new_batch_state = (state, rng)
                return new_batch_state, test_loss_info
            
            print("Training epoch")
            (
                train_state,
                rng,
            ) = update_state
            # prepare training batchs
            obs, action, world_state, done, _, _, rewards, seq_lens = dataset
            batched_trajs = Trajectory(   # from baseline
                obs=obs,    # (data_size, max_seq_len, obs_dim)
                action=action.squeeze(),  # (data_size, max_seq_len)
                world_state=world_state,    # (data_size, max_seq_len, world_state_dim)
                done=done,  # (data_size, max_seq_len)
            )
            max_reward_diff = jnp.max(rewards) - jnp.min(rewards)
            rewards = jnp.expand_dims(rewards, axis=-1)
            seq_lens = jnp.expand_dims(seq_lens, axis=-1)
            inith_state = ScannedRNN.initialize_carry(train_data_size, config["RNN_HIDDEN_SIZE"])
            batch = (
                inith_state,    # (data_size, hidden_size)
                batched_trajs,  # key: (data_size, max_seq_len, other_dim)
                rewards,    # (data_size, 1)
                seq_lens,   # (data_size, 1)
            )

            permutation = jax.random.permutation(rng, train_data_size)
            shuffled_batch = jax.tree_util.tree_map(lambda x: jnp.take(x, permutation, axis=0), batch)

            minibatches = jax.tree_util.tree_map(
                lambda x: jnp.swapaxes(
                        jnp.reshape(
                        x,
                        [config["REWARD_MODEL_MINIBATCHES"], config["MINIBATCH_SIZE"], -1] + list(x.shape[2:]),
                    ),
                    2, 1,
                ),
                shuffled_batch,
            )
            rng, _rng = jax.random.split(rng)
            minibatch_state = (train_state, _rng)
            minibatch_state, loss_info = jax.lax.scan(train_minibatch, minibatch_state, minibatches)
            # prepare test batchs
            obs, action, world_state, done, _, _, rewards, seq_lens = testset
            batched_test_trajs = Trajectory(   # from baseline
                obs=obs,    # (data_size, max_seq_len, obs_dim)
                action=action.squeeze(),  # (data_size, max_seq_len)
                world_state=world_state,    # (data_size, max_seq_len, world_state_dim)
                done=done,  # (data_size, max_seq_len)
            )
            rewards = jnp.expand_dims(rewards, axis=-1)
            seq_lens = jnp.expand_dims(seq_lens, axis=-1)
            
            test_batch = (
                ScannedRNN.initialize_carry(test_data_size, config["RNN_HIDDEN_SIZE"]),    # (data_size, hidden_size)
                batched_test_trajs,  # key: (data_size, max_seq_len, other_dim)
                rewards,    # (data_size, 1)
                seq_lens,   # (data_size, 1)
            )
            test_permutation = jax.random.permutation(rng, test_data_size)
            shuffled_test_batch = jax.tree_util.tree_map(lambda x: jnp.take(x, test_permutation, axis=0), test_batch)
            test_minibatches = jax.tree_util.tree_map(
                lambda x: jnp.swapaxes(
                        jnp.reshape(
                        x,
                        [config["REWARD_MODEL_TEST_MINIBATCHES"], config["MINIBATCH_SIZE"], -1] + list(x.shape[2:]),
                    ),
                    2, 1,
                ),
                shuffled_test_batch,
            )
            minibatch_state, test_loss_info = jax.lax.scan(test_minibatch, minibatch_state, test_minibatches)
            
            train_state, rng = minibatch_state
            update_state = (
                train_state,
                rng)
            total_loss = jnp.mean(loss_info["total_loss"])
            value_loss = jnp.mean(loss_info["value_loss"])
            mse_loss = jnp.mean(loss_info["mse_loss"])
            baseline_loss = jnp.mean(loss_info["baseline_loss"])
            test_total_loss = jnp.mean(test_loss_info["total_loss"])
            test_value_loss = jnp.mean(test_loss_info["value_loss"])

            # save model every 1000 epochs
            update_epochs += 1
            def callback(params, update_epochs, total_loss, value_loss, mse_loss, best_test_loss, baseline_loss, test_total_loss, test_value_loss):
                

                wandb.log({
                    "reward model loss": total_loss,
                    "value loss": value_loss,
                    "mse loss": mse_loss,
                    "learning rate": lr_schedule(update_epochs),
                    "reward model test loss": test_total_loss,
                    "test value loss": test_value_loss,
                    }, step=int(update_epochs))
                print("Epoch: ", update_epochs, "value Loss: ", value_loss, "test value loss: ", test_value_loss)
                print("Total Training Loss: ", total_loss, "Total Test Loss", test_total_loss, " MSE Loss: ", mse_loss, " Baseline Loss: ", baseline_loss)
                if test_total_loss < best_test_loss:
                    model_name = f"reward_model_best.msgpack"
                    save_model(params, config["SAVE_REWARD_MODEL_PATH"], model_name)
                if update_epochs % 1000 != 0:
                    return
                if not os.path.exists(config["SAVE_REWARD_MODEL_PATH"]):
                    os.makedirs(config["SAVE_REWARD_MODEL_PATH"])
                model_name = f"reward_model_{update_epochs}.msgpack"
                save_model(params, config["SAVE_REWARD_MODEL_PATH"], model_name)

            jax.experimental.io_callback(callback, None, train_state.params, update_epochs, total_loss, value_loss, mse_loss, best_test_loss, baseline_loss, test_total_loss, test_value_loss)
            best_test_loss = jnp.min(jnp.array([test_total_loss, best_test_loss]))
            
            return (update_state, dataset, testset, update_epochs, best_test_loss), (total_loss, test_total_loss)

        train_epoch = jax.jit(train_epoch)
        update_runner_state = (update_state, dataset, testset, 0, 1.0)
        print("Training reward model")
        update_runner_state, loss = jax.lax.scan(train_epoch, update_runner_state, None, config["REWARD_MODEL_EPOCHS"])
        update_state, dataset, testset, update_epochs, best_test_loss = update_runner_state
        total_loss, test_total_loss = loss

        print("Training finished after ", update_epochs, " epochs")
        print("Training loss: ", total_loss)
        print("Test loss: ", test_total_loss)
        print("Best test loss: ", best_test_loss)

        return update_state, loss
    
    return train


def train_model(config):
    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    if config['ENV_NAME'] == "MPE_simple_tag_v3":
        env = CooperativeEnvWrapperLoaded(env)
    action_dim = env.action_space(env.agents[0]).n
    if config["REWARD_MODEL_TYPE"] == "FF":
        model = RewardModelFF(action_dim=action_dim, layer_dim=config["FF_LAYER_DIM"])
    elif config["REWARD_MODEL_TYPE"] == "RNN":
        config["RNN_HIDDEN_SIZE"] = config["FF_LAYER_DIM"]
        model = RewardModel(hidden_size=config["RNN_HIDDEN_SIZE"], action_dim=action_dim)
    rng = jax.random.PRNGKey(config["SEED"])

    # DATALOADER
    if config["VDN"]:
        data_filename_list = []
        if config["ENV_NAME"] == "MPE_simple_spread_v3":
            if env.num_agents == 3:
                performance_list = ['149', '198', '297', '348', '387']
                if config["USE_UNILATERAL"]:
                    unilateral_performance_list = ['149|198unilateral', '149|297unilateral', '149unilateral']
            elif env.num_agents == 4:
                performance_list = ['99', '118', '136', '150', '175']
                if config["USE_UNILATERAL"]:
                    unilateral_performance_list = ['99|136unilateral', '99|150unilateral', '99|175unilateral']
            elif env.num_agents == 5:
                performance_list = ['99', '117', '153', '165', '184']
                if config["USE_UNILATERAL"]:
                    unilateral_performance_list = ['99|153unilateral', '99|165unilateral', '99|184unilateral']
            elif env.num_agents == 6:
                performance_list = ['137', '157', '176', '190', '200']
                if config["USE_UNILATERAL"]:
                    unilateral_performance_list = ['137|176unilateral', '137|190unilateral', '137|200unilateral']
            elif env.num_agents == 7:
                performance_list = ['159', '169', '177', '195', '216']
                if config["USE_UNILATERAL"]:
                    unilateral_performance_list = ['159|177unilateral', '159|195unilateral', '159|216unilateral']
        elif config["ENV_NAME"] == "MPE_simple_reference_v3":
            performance_list = ['104', '149', '199', '242', '298']
            if config["USE_UNILATERAL"]:
                unilateral_performance_list = ['104|149unilateral', '104|199unilateral', '104unilateral']
        elif config["ENV_NAME"] == "MPE_simple_tag_v3":
            performance_list = ['621', '525', '392', '271', '187']
            if config["USE_UNILATERAL"]:
                unilateral_performance_list = ['621|480unilateral', '621|271unilateral', '621|187unilateral']

        print("Length of num_batches", len(config["NUM_BATCH"]))
        for i in range(len(config["NUM_BATCH"])):
            if config["ENV_NAME"] == "MPE_simple_spread_v3" and env.num_agents != 3:
                data_filename_list += ["na" + str(env.num_agents) + "/vdn" + str(performance_list[i]) + "/traj_batch_" + str(x) + ".pkl" for x in range(0, config["NUM_BATCH"][i])]
            else:
                data_filename_list += ["vdn" + str(performance_list[i]) + "/traj_batch_" + str(x) + ".pkl" for x in range(0, config["NUM_BATCH"][i])]
        if config["USE_UNILATERAL"]:
            for i in range(len(config["NUM_UNILATERAL_BATCH"])):
                if config["ENV_NAME"] == "MPE_simple_spread_v3" and env.num_agents != 3:
                    data_filename_list += ["na" + str(env.num_agents) + "/vdn" + str(unilateral_performance_list[i]) + "/traj_batch_" + str(x) + ".pkl" for x in range(0, config["NUM_UNILATERAL_BATCH"][i])]
                else:
                    data_filename_list += ["vdn" + str(unilateral_performance_list[i]) + "/traj_batch_" + str(x) + ".pkl" for x in range(0, config["NUM_UNILATERAL_BATCH"][i])]
        dir_paths = config["DATA_DIR"]
    else:
        data_filename_list = config["CONVERTED_FILELIST"]
        dir_paths = config["CONVERTED_DIR"]
    data_filepath_list = [os.path.join(dir_path, dir_name) for dir_path in dir_paths for dir_name in data_filename_list]
        
    # JAX DATALOADER
    print("Initializing JaxDataLoader")
    dataloader = JaxDataLoader(
        dir_path="",
        file_list=data_filepath_list,
        env=env,
        seed=config["SEED"],
        vdn=config["VDN"],
        debug=config["DEBUG"],
        batch_size=config["MINIBATCH_SIZE"]
    )
    print("JaxDataLoader initialized")

    # with jax.disable_jit(True):
    with jax.disable_jit(config["DISABLE_JIT"]):
        train_jit = make_train_model(model, dataloader, config)
        update_state, loss = train_jit(rng)


@hydra.main(version_base=None, config_path="config", config_name="rm_config")
def main(config):
    config = OmegaConf.to_container(config)
    print(jax.devices())
    if config["DEBUG"]:
        print("DEBUG MODE")
        config["DISABLE_JIT"] = True
    config["PROJECT"] = config["ENV_NAME"] + "_reward_model"
    if config["ENV_NAME"].split("_")[0] == "MPE":
        env_name_str = config["ENV_NAME"].split("_")[2]
    else:
        env_name_str = config["ENV_NAME"]
        
    config["DATA_DIR"][0] = f'data/{env_name_str}/'
    if env_name_str == "spread":
        if config["USE_UNILATERAL"]:
            config["SAVE_REWARD_MODEL_PATH"] = f'results/{env_name_str}/na{config["ENV_KWARGS"]["num_agents"]}/unilateral/rm_{config["REWARD_MODEL_TYPE"]}_{config["FF_LAYER_DIM"]}_mse_{config["MSE_LOSS_COEF"]}_vdn_' + '|'.join([str(x) for x in config['NUM_BATCH']]) + "_" + '|'.join([str(x) for x in config['NUM_UNILATERAL_BATCH']])
        else:
            config["SAVE_REWARD_MODEL_PATH"] = f'results/{env_name_str}/na{config["ENV_KWARGS"]["num_agents"]}/rm_{config["REWARD_MODEL_TYPE"]}_{config["FF_LAYER_DIM"]}_mse_{config["MSE_LOSS_COEF"]}_vdn_' + '|'.join([str(x) for x in config['NUM_BATCH']])
    else:
        if config["USE_UNILATERAL"]:
            config["SAVE_REWARD_MODEL_PATH"] = f'results/{env_name_str}/unilateral/rm_{config["REWARD_MODEL_TYPE"]}_{config["FF_LAYER_DIM"]}_mse_{config["MSE_LOSS_COEF"]}_vdn_' + '|'.join([str(x) for x in config['NUM_BATCH']]) + "_" + '|'.join([str(x) for x in config['NUM_UNILATERAL_BATCH']])
        else:
            config["SAVE_REWARD_MODEL_PATH"] = f'results/{env_name_str}/rm_{config["REWARD_MODEL_TYPE"]}_{config["FF_LAYER_DIM"]}_mse_{config["MSE_LOSS_COEF"]}_vdn_' + '|'.join([str(x) for x in config['NUM_BATCH']])

    wandb_name = f'RM_mse{config["MSE_LOSS_COEF"]}_vdn_' + '|'.join([str(x) for x in config['NUM_BATCH']]) + "_" + '|'.join([str(x) for x in config['NUM_UNILATERAL_BATCH']]) if config["USE_UNILATERAL"] else ''
    wandb.init(
        entity=config["ENTITY"],
        project=str(config["PROJECT_PREFIX"]) + config["PROJECT"],
        tags=["REWARD_MODEL", config["REWARD_MODEL_TYPE"], config["ENV_NAME"]],
        config=config,
        mode=config["WANDB_MODE"],
        name=wandb_name
    )

    train_model(config)

if __name__ == "__main__":
    main()