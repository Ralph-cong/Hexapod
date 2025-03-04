from stable_baselines3 import PPO
import torch
import os
import tyro

from stable_baselines3.common.env_util import make_vec_env
from hexapod_env import HexapodCPGEnv


def main(from_scratch: bool = False, steps: int = 80000, model_path: str = "checkpoints"):
    # # 将环境包装成向量化环境，以支持更快的训练
    # env = make_vec_env(lambda: HexapodCPGEnv(render_mode='human'), n_envs=1)
    env = make_vec_env(lambda: HexapodCPGEnv(
        render_mode='rgb_array'), n_envs=2)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(model_path, exist_ok=True)
    save_path = os.path.join(model_path, "hexapod_cpg_ppo")

    # 如果从头训练，则创建新模型；否则加载已有模型
    if from_scratch:
        model = PPO("MlpPolicy", env, verbose=1, device=device,
                    learning_rate=0.001, gamma=0.9, ent_coef=0.001, tensorboard_log="./Tensorboard/")
    else:
        model = PPO.load(save_path, env=env, device=device)

    model.learn(total_timesteps=steps,
                reset_num_timesteps=False)  # Optional | tb_log_name="PPO"

    model.save(save_path)
    env.close()


if __name__ == "__main__":
    tyro.cli(main)
