from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO
from hexapod_env import HexapodCPGEnv
import gymnasium as gym
import os


def main():
    # 加载模型进行测试
    env = make_vec_env(lambda: HexapodCPGEnv(render_mode='human'), n_envs=1)

    save_path = os.path.join("checkpoints", "hexapod_cpg_ppo")
    model = PPO.load(save_path, env=env)

    # 评估模型
    mean_reward, std_reward = evaluate_policy(
        model, model.get_env(), n_eval_episodes=10)
    print(f"Mean reward: {mean_reward}, Std reward: {std_reward}")

    # 关闭环境
    env.close()


if __name__ == "__main__":
    main()
