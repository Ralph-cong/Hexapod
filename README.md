# Reinforcement Learning Gait Control for hexapod robots

## Installation
```shell
conda create -n "hexapod" python=3.10
```

Stable-Baselines3 requires python 3.9+ and PyTorch >= 2.3
```shell
pip3 install torch torchvision torchaudio
pip install stable-baselines3[extra] 
pip install pybullet tyro tensorboard==2.17.0
```


## Tips
- The env is wrapped in `hexapod_env.py`
- command for train `python train.py [--from_scratch True --steps 50000 --model_path "checkpoints"]`
- There are different versions of CPG in dir `CPGs`
- *Enjoy your journey in RL*