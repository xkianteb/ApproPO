# Reinforcement Learning with Convex Constraints

PyTorch implementation of the paper:

[Reinforcement Learning with Convex Constraints](https://papers.nips.cc/paper/9556-reinforcement-learning-with-convex-constraints.pdf)\
Sobhan Miryoosefi, Kianté Brantley, Hal Daumé III, Miroslav Dudik, Robert Schapire\
NeurIPS 2019 

```bash
python setup.py develop
```
# Build guide
###1. Create conda environment
<code>
conda create -n approPO 

pip install -r requirements.txt
</code>

###2. Run experiments
- Config file: args.py

- Paper setting: 
<code> python approPO.py </code>

- RCPO:
<code> python rcpo.py </code>