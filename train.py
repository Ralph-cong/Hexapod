from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
import torch
import os
import tyro
from hexapod_env import HexapodCPGEnv
import time


def main(from_scratch: bool = True, steps: int = 500_000, model_path: str = "checkpoints"):

    n_envs = 1
    env = make_vec_env(lambda: HexapodCPGEnv(render_mode='human'), n_envs=n_envs)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(model_path, exist_ok=True)
    save_path = os.path.join(model_path, "hexapod_cpg_ppo")

    # 设置策略网络结构
    policy_kwargs = dict(net_arch=[256, 256, 256, 256])

    if from_scratch:
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            device=device,
            learning_rate=1e-4,
            gamma=0.99,
            ent_coef=0.001,
            n_steps=512 // n_envs,
            gae_lambda=0.95,
            clip_range=0.2,
            policy_kwargs=policy_kwargs,
            tensorboard_log="./Tensorboard/"
        )
    else:
        model = PPO.load(save_path, env=env, device=device)

    # 创建一个保存 checkpoint 的回调，每20分钟（1200秒）保存一次
    checkpoint_callback = CheckpointCallback(
        save_freq=1,  # 实际时间间隔由 custom callback 控制
        save_path=model_path,
        name_prefix="ppo_checkpoint",
        save_replay_buffer=False,
        save_vecnormalize=False
    )

    # 包装成按时间控制的 callback
    class TimeCheckpointWrapper:
        def __init__(self, callback, interval_seconds=1200):
            self.callback = callback
            self.interval = interval_seconds
            self.last_time = time.time()

        def __call__(self, locals_, globals_):
            current_time = time.time()
            if current_time - self.last_time >= self.interval:
                print(f"\n[Checkpoint] Saving model at {time.strftime('%H:%M:%S')}")
                self.callback._on_step()
                self.last_time = current_time
            return True

    # 训练模型并定期保存
    model.learn(
        total_timesteps=steps,
        reset_num_timesteps=False,
        callback=TimeCheckpointWrapper(checkpoint_callback)
    )

    # 保存最终模型
    model.save(save_path)
    env.close()


if __name__ == "__main__":
    tyro.cli(main)
