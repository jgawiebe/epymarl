# Fork of EPyMARL for use with CybORG reinforcement learning environment

*In development*: CybORG is currently available for a single RL agent. The pyMARLWrapper class wraps the environment so that it can be trained on using EPyMARL's multi-agent gym functionality.

# Training a CybORG Agent
`python src\\main.py --config=mappo_ns --env-config=gymma with env_args.time_limit=100 env_args.key="cyborg" t_max=2000000`

[CybORG](https://github.com/cage-challenge/cage-challenge-2) (Cyber Operations Research Gym) is a research environemnt for training autonomous agents.

EPyMARL is  an extension of [PyMARL](https://github.com/oxwhirl/pymarl), and includes
- Additional algorithms (IA2C, IPPO, MADDPG, MAA2C and MAPPO)
- Support for [Gym](https://github.com/openai/gym) environments (on top of the existing SMAC support)
- Option for no-parameter sharing between agents (original PyMARL only allowed for parameter sharing)
- Flexibility with extra implementation details (e.g. hard/soft updates, reward standarization, and more)
- Consistency of implementations between different algorithms (fair comparisons)

# Run an experiment on a Gym environment

```shell
python3 src/main.py --config=qmix --env-config=gymma with env_args.time_limit=50 env_args.key="lbforaging:Foraging-8x8-2p-3f-v1"
```
 In the above command `--env-config=gymma` (in constrast to `sc2` will use a Gym compatible wrapper). `env_args.time_limit=50` sets the maximum episode length to 50 and `env_args.key="..."` provides the Gym's environment ID. In the ID, the `lbforaging:` part is the module name (i.e. `import lbforaging` will run automatically).


The config files act as defaults for an algorithm or environment. 

They are all located in `src/config`.
`--config` refers to the config files in `src/config/algs`
`--env-config` refers to the config files in `src/config/envs`

All results will be stored in the `Results` folder.

# Run a hyperparameter search

We include a script named `search.py` which reads a search configuration file (e.g. the included `search.config.example.yaml`) and runs a hyperparameter search in one or more tasks. The script can be run using
```shell
python search.py run --config=search.config.example.yaml --seeds 5 locally
```
In a cluster environment where one run should go to a single process, it can also be called in a batch script like:
```shell
python search.py run --config=search.config.example.yaml --seeds 5 single 1
```
where the 1 is an index to the particular hyperparameter configuration and can take values from 1 to the number of different combinations.

# Saving and loading learnt models

## Saving models

You can save the learnt models to disk by setting `save_model = True`, which is set to `False` by default. The frequency of saving models can be adjusted using `save_model_interval` configuration. Models will be saved in the result directory, under the folder called *models*. The directory corresponding each run will contain models saved throughout the experiment, each within a folder corresponding to the number of timesteps passed since starting the learning process.

## Loading models

Learnt models can be loaded using the `checkpoint_path` parameter, after which the learning will proceed from the corresponding timestep. 

# Citing PyMARL and EPyMARL

The Extended PyMARL (EPyMARL) codebase was used in [Benchmarking Multi-Agent Deep Reinforcement Learning Algorithms in Cooperative Tasks](https://arxiv.org/abs/2006.07869).

*Georgios Papoudakis, Filippos Christianos, Lukas Schäfer, & Stefano V. Albrecht. Benchmarking Multi-Agent Deep Reinforcement Learning Algorithms in Cooperative Tasks, Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks (NeurIPS), 2021*

In BibTeX format:

```tex
@inproceedings{papoudakis2021benchmarking,
   title={Benchmarking Multi-Agent Deep Reinforcement Learning Algorithms in Cooperative Tasks},
   author={Georgios Papoudakis and Filippos Christianos and Lukas Schäfer and Stefano V. Albrecht},
   booktitle = {Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks (NeurIPS)},
   year={2021},
   url = {http://arxiv.org/abs/2006.07869},
   openreview = {https://openreview.net/forum?id=cIrPX-Sn5n},
   code = {https://github.com/uoe-agents/epymarl},
}
```

If you use the original PyMARL in your research, please cite the [SMAC paper](https://arxiv.org/abs/1902.04043).

*M. Samvelyan, T. Rashid, C. Schroeder de Witt, G. Farquhar, N. Nardelli, T.G.J. Rudner, C.-M. Hung, P.H.S. Torr, J. Foerster, S. Whiteson. The StarCraft Multi-Agent Challenge, CoRR abs/1902.04043, 2019.*

In BibTeX format:

```tex
@article{samvelyan19smac,
  title = {{The} {StarCraft} {Multi}-{Agent} {Challenge}},
  author = {Mikayel Samvelyan and Tabish Rashid and Christian Schroeder de Witt and Gregory Farquhar and Nantas Nardelli and Tim G. J. Rudner and Chia-Man Hung and Philiph H. S. Torr and Jakob Foerster and Shimon Whiteson},
  journal = {CoRR},
  volume = {abs/1902.04043},
  year = {2019},
}
```

# License
All the source code that has been taken from the PyMARL repository was licensed (and remains so) under the Apache License v2.0 (included in `LICENSE` file).
Any new code is also licensed under the Apache License v2.0
