# Empirical Analysis of Automated Stock Trading Using Deep Reinforcement Learning
This repository offers codes for [MDPI Applied Sciences](https://www.mdpi.com/2076-3417/13/1/633#B10-applsci-13-00633)

## Ubuntu
### Prerequisites
```bash
sudo apt-get update && sudo apt-get install cmake libopenmpi-dev python3-dev zlib1g-dev libgl1-mesa-glx
```
### Conda Virtual Environment Setting
The requirements.txt file contains only the most basic libraries necessary for running the model. You may need to install additional libraries.
```bash
conda create -n <env_name> python==3.7
conda activate <env_name>
cd EAASTUDRL
pip install -r requirements.txt
```
### Run
```bash
cd EAASTUDRL
python run_DRL_dji.py
# python run_DRL_kospi.py
# python run_DRL_tse.py
```

## Baseline Reference
### Paper
[Yang, Hongyang, et al. "Deep reinforcement learning for automated stock trading: An ensemble strategy." Proceedings of the first ACM international conference on AI in finance. 2020.](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3690996) 
### Github
[<img src="https://img.shields.io/badge/Github-222222?style=flate&logo=Github&logoColor=white"/>](https://github.com/AI4Finance-Foundation/FinRL-Live-Trading)

