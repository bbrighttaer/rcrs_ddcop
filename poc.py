import random
from logging import getLogger
from time import perf_counter

import numpy as np
import xgboost as xgb
import torch
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from skopt.space import Real, Categorical, Integer
from skopt import gp_minimize

from rcrs_ddcop.core.experience import ExperienceBuffer
from rcrs_ddcop.core.nn import ModelTrainer, NodeGCN, XGBTrainer

seed = 7
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

data = torch.load('AmbulanceTeamAgent_482151809.pt')
exp_buffer = ExperienceBuffer()
exp_buffer.memory = data

mode = 'xgb_train'

sim_time_steps = 100

if mode == 'train':
    trainer = ModelTrainer(
        model=NodeGCN(dim=7),
        label='poc-agent-trainer',
        experience_buffer=exp_buffer,
        log=getLogger(__name__),
        transform=StandardScaler(),
        batch_size=32,
        lr=6e-3,  # 0.006060360647573297,
        epochs=100,
        weight_decay=0,
    )

    start = perf_counter()
    for i in range(sim_time_steps):
        trainer()
    print(f'time = {perf_counter() - start}')

elif mode == 'xgb_train':
    trainer = XGBTrainer(
        label='poc-agent-xgb-trainer',
        experience_buffer=exp_buffer,
        log=getLogger(__name__),
        transform=StandardScaler(),
        rounds=100,
        model_params={
            'objective': 'reg:squarederror',
            'max_depth': 10,
            'learning_rate': 1e-2,
            'reg_lambda': 1e-3,
        }
    )

    start = perf_counter()
    for i in range(10):
        score = trainer()
    print(f'time = {perf_counter() - start}, score = {score}')
else:
    def objective_function(args):
        lr, epochs, batch_size, weight_decay = float(args[0]), int(args[1]), int(args[2]), float(args[3])
        print(f'lr={lr}, epochs={epochs}, batch_size={batch_size}, weight_decay={weight_decay}')

        trainer = ModelTrainer(
            model=NodeGCN(dim=7),
            label='poc-agent-trainer',
            experience_buffer=exp_buffer,
            log=getLogger(__name__),
            transform=StandardScaler(),
            batch_size=batch_size,
            lr=lr,
            epochs=epochs,
            weight_decay=weight_decay,
        )
        s = []
        for i in range(sim_time_steps):
            s.append(trainer())
        score = np.mean(s)
        return score


    hparam_search = {
        'lr': Real(1e-6, 1e-1, prior='log-uniform'),
        'epochs': Integer(4, 100),
        'batch_size': Categorical([4, 16, 32]),
        'weight_decay': Real(1e-6, 1e-1),
    }

    res = gp_minimize(objective_function, hparam_search.values(), n_calls=100)
    print(f'Best score={res.fun:.4f}, lr={res.x[0]}, epochs={res.x[1]}, batch_size={res.x[2]}, weigh_decay={res.x[3]}')
