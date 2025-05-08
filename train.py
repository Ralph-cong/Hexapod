from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
import torch
import os
import tyro
from hexapod_env import HexapodCPGEnv
import time


class TimeBasedCheckpointCallback(BaseCallback):
    def __init__(self, save_path: str, interval_seconds: int = 1200, name_prefix: str = "ppo_checkpoint", verbose: int = 1):
        super().__init__(verbose)
        self.save_path = save_path
        self.interval_seconds = interval_seconds
        self.name_prefix = name_prefix
        self.last_checkpoint_time = time.time()
        os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        current_time = time.time()
        if current_time - self.last_checkpoint_time >= self.interval_seconds:
            checkpoint_file = os.path.join(
                self.save_path,
                f"{self.name_prefix}_{self.num_timesteps}_steps"
            )
            self.model.save(checkpoint_file)
            if self.verbose:
                print(f"\n[Checkpoint] Saved model to {checkpoint_file} at {time.strftime('%H:%M:%S')}")
            self.last_checkpoint_time = current_time
        return True


def main(from_scratch: bool = True, steps: int = 500_000, model_path: str = "checkpoints"):

    n_envs = 1
    env = make_vec_env(lambda: HexapodCPGEnv(render_mode='none'), n_envs=n_envs)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(model_path, exist_ok=True)
    save_path = os.path.join(model_path, "hexapod_cpg_ppo")

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

    # 使用基于时间的 checkpoint callback
    checkpoint_callback = TimeBasedCheckpointCallback(
        save_path=model_path,
        interval_seconds=200,
        name_prefix="ppo_checkpoint"
    )

    model.learn(
        total_timesteps=steps,
        reset_num_timesteps=False,
        callback=checkpoint_callback
    )

    # 保存最终模型
    model.save(save_path)
    env.close()


if __name__ == "__main__":
    tyro.cli(main)
