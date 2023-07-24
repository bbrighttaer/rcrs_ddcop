import random
from logging import getLogger
from time import perf_counter

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from rcrs_ddcop.core.experience import ExperienceBuffer
from rcrs_ddcop.core.nn import ModelTrainer, NodeGCN

seed = 7
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


# data = torch.load('210552869.pt')
data = torch.load('1962675462.pt')
exp_buffer = ExperienceBuffer()
exp_buffer.memory = data


trainer = ModelTrainer(
    model=NodeGCN(dim=7),
    label='poc-agent-trainer',
    experience_buffer=exp_buffer,
    log=getLogger(__name__),
    transform=StandardScaler(),
    batch_size=16,
    lr=1e-3,
)

start = perf_counter()
for i in range(1000):
    trainer()
print(f'time = {perf_counter() - start}')
