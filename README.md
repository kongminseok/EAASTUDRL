<div align="center">

<h1>Empirical Analysis of Automated Stock Trading Using Deep Reinforcement Learning</h1>

<div>
    <a href='https://github.com/kongminseok' target='_blank'>Minseok Kong</a><sup>1</sup>&emsp;
    <a href='https://icslsogang.github.io/' target='_blank'>Jungmin So</a><sup>2</sup>&emsp;
</div>

<div>
    <sup>1</sup>Sogang University&emsp; 
</div>

<div>
    <h4 align="center">
        • <a href="https://www.mdpi.com/2076-3417/13/1/633#B10-applsci-13-00633" target='_blank'>[paper]</a> •
    </h4>
</div>

</div>


## Ubuntu Installation
### Prerequisites
```bash
sudo apt-get update && sudo apt-get install cmake libopenmpi-dev python3-dev zlib1g-dev libgl1-mesa-glx
```
### Conda Virtual Environment Setting
The requirements.txt file contains only the most basic libraries necessary for running the model. You may need to install additional libraries.
```bash
conda create -n <env_name> python=3.7
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

## Acknowledgement
This implementation is based on a [paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3690996) and [GitHub repo](https://github.com/AI4Finance-Foundation/FinRL-Live-Trading). Thanks for the great work.
