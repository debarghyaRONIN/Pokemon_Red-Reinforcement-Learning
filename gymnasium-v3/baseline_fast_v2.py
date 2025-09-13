import sys
import glob
from os.path import exists
from pathlib import Path
from red_gym_env_v2 import RedGymEnv
from stream_agent_wrapper import StreamWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from tensorboard_callback import TensorboardCallback

def make_env(rank, env_conf, seed=0):
    """ Utility function for multiprocessed env. """
    def _init():
        env = StreamWrapper(
            RedGymEnv(env_conf), 
            stream_metadata={
                "user": "DebarghyaSaha",
                "env_id": rank,
                "color": "#447799",
                "extra": "",
            }
        )
        env.reset(seed=(seed + rank))
        return env
    return _init

def find_latest_checkpoint(checkpoint_dir, prefix="poke"):
    """ Finds the latest saved checkpoint file. """
    checkpoints = glob.glob(f"{checkpoint_dir}/{prefix}_*.zip")
    if checkpoints:
        try:
            return max(checkpoints, key=lambda x: int(x.split('_')[-2]))  # Sorting based on step count
        except ValueError:
            return None  # Handle cases where filenames are malformed
    return None

if __name__ == "__main__":
    use_wandb_logging = False
    ep_length = 2048 * 80
    sess_id = "runs"
    sess_path = Path(sess_id)
    num_cpu = 64  # Number of parallel environments

    env_config = {
        'headless': False, 'save_final_state': True, 'early_stop': False,
        'action_freq': 10, 'init_state': '../has_pokedex_nballs.state', 'max_steps': ep_length, 
        'print_rewards': True, 'save_video': False, 'fast_video': True, 'session_path': sess_path,
        'gb_path': '../PokemonRed.gb', 'debug': True, 'reward_scale': 0.5, 'explore_weight': 0.25
    }
    
    print("Environment Configuration:", env_config)

    set_random_seed(0)
    env = SubprocVecEnv([make_env(i, env_config) for i in range(num_cpu)])
    
    checkpoint_callback = CheckpointCallback(save_freq=ep_length//2, save_path=sess_path, name_prefix="poke")
    callbacks = [checkpoint_callback, TensorboardCallback(sess_path)]

    if use_wandb_logging:
        import wandb
        from wandb.integration.sb3 import WandbCallback
        
        with wandb.init(
            project="pokemon-train",
            id=sess_id,
            name="v2-a",
            config=env_config,
            sync_tensorboard=True,  
            monitor_gym=True,  
            save_code=True,
        ) as run:
            wandb.tensorboard.patch(root_logdir=str(sess_path))
            callbacks.append(WandbCallback())

    # Find the latest checkpoint
    latest_checkpoint = find_latest_checkpoint(sess_path)
    train_steps_batch = ep_length // 64
    
    if latest_checkpoint:
        print(f"\nLoading latest checkpoint: {latest_checkpoint}")
        model = PPO.load(latest_checkpoint, env=env)
        model.n_steps = train_steps_batch
        model.n_envs = num_cpu
        model.rollout_buffer.buffer_size = train_steps_batch
        model.rollout_buffer.n_envs = num_cpu
        model.rollout_buffer.reset()
    else:
        print("\nNo checkpoint found. Training from scratch...")
        model = PPO("MultiInputPolicy", env, verbose=1, 
            n_steps=16384, batch_size=8192,  # Increased for better GAE approximation
            n_epochs=10, gamma=0.999, ent_coef=0.01, 
            gae_lambda=0.98,  # High lambda for near-complete GAE
            learning_rate=2.5e-4, tensorboard_log=str(sess_path), 
            device='cuda')


    print("Policy Details:\n", model.policy)
    
    model.learn(total_timesteps=ep_length * num_cpu * 10000, callback=CallbackList(callbacks), tb_log_name="poke_ppo")
    
    if use_wandb_logging:
        run.finish()
