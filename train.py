import torch
import torch_geometric
import logging
from pathlib import Path
from tqdm import tqdm
import os
from ocpmodels import models
from ocpmodels.common import logger
from ocpmodels.common.utils import setup_logging, load_config
from ocpmodels.datasets import LmdbDataset
from ocpmodels.common.registry import registry
from ocpmodels.trainers import EnergyTrainer, ForcesTrainer
setup_logging()
from pyparsing import identbodychars
from torch import seed



torch.cuda.empty_cache()
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
#conf=load_config(config_path)[0]
task={'dataset': 'single_point_lmdb',
 'description': 'Regressing to energies for DFT Iron structures',
 'type': 'regression',
 'metric': 'mae',
 'labels': ['potential energy'],
 }

model={'name': 'SchNet_EwaldMP_Q',
 'hidden_channels': 16,
 'num_filters': 16,
 'num_interactions': 5,
 'num_gaussians': 20,
 'cutoff': 6.0,
 'max_neighbors': 50,
 'use_pbc': False,
 'otf_graph': True,
 'regress_forces': False,
 'readout': 'add',
 'seperated': True,
 'residual': True,
  'ewald_hyperparams': {'k_cutoff': 0.1,'delta_k':0.2,'num_k_rbf':4,'downprojection_size':8,'num_hidden':2}}

optimizer={'batch_size': 64,
 'eval_batch_size': 32,
 'num_workers': 4,
 'lr_initial': 0.0005,
 'lr_gamma': 0.1,
 'lr_milestones': [6000, 8000, 10000],
 'warmup_steps': 1000,
 'warmup_factor': 0.2,
 'max_epochs': 250,}

name='schnet_zinc'

logger='wandb'

dataset=[{'src': 'data/zinc_train_31611.lmdb',
  'normalize_labels': True,
  'target_mean': -31.2988981640534,
  'target_std': 5.383123813548416} ,
 {'src': 'data/zinc_val_4000.lmdb'},
 {'src': 'data/zinc_test_4000.lmdb'}]


trainer=EnergyTrainer(task=task,
                      model=model,
                      dataset=dataset,
                      optimizer=optimizer,
                      identifier=name,
                      run_dir='runs',
                      is_debug=False,
                      print_every=1000,
                      seed=42,
                      logger=logger,
                      local_rank=0,
                      amp=False)
trainer.train()