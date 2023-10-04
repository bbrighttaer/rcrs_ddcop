import random
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
from circular_dict import CircularDict
from torch_geometric.data import Data


@dataclass
class Experience:
    state: List
    action: List
    utility: float
    next_state: Optional[List]

    def to_dict(self):
        return self.__dict__


def process_data(raw_data: list[Data]) -> list:
    state = raw_data[0]
    s_prime = raw_data[1]

    # remove uninformative rows
    # idx = state.x[:, 0] != s_prime.x[:, 0]
    # state.x = state.x[idx]
    # s_prime.x = s_prime.x[idx]
    # if len(state.x) == 0 or len(s_prime.x) == 0:
    #     return []

    # identify change in fieriness
    # diff = torch.clip(state.x[:, 0] - s_prime.x[:, 0], 0, 1).view(-1, 1)
    # x = torch.concat([state.x[:, :-1], diff], dim=1)
    x = torch.concat([state.x, s_prime.x], dim=1).tolist()
    return x


class ExperienceBuffer:
    """Stores data for training models"""

    def __init__(self, log, capacity: int = 1000, lbl: str = ''):
        self.memory = CircularDict(maxlen=capacity)
        self.lbl = lbl
        self.log = log
        self.ref_data_sz = 100
        self.sim_threshold = 0.5

    def add(self, experience: List[Data]) -> list:
        """Adds an experience to the buffer"""
        x = process_data(experience)
        sz = len(self)
        keys = [f'{self.lbl}_{sz + i}' for i, e in enumerate(x)]
        if experience and x:
            self.merge_experiences(x, keys)
        return keys

    def add_processed_exp(self, exp, key):
        self.memory[key] = exp

    def get_keys(self):
        return list(self.memory.keys())

    def sample(self, batch_size: int) -> List:
        """
        Samples experiences from the buffer.

        :param batch_size: the number of samples to take from the buffer. If batch size is less than
                           the size of the buffer, a ValueError is thrown.
        :return: list of experiences.
        """
        if batch_size < len(self):
            return random.sample(list(self.memory.values()), batch_size)
        elif self.memory:
            return random.sample(list(self.memory.values()), len(self))

    def select_exps_by_key(self, keys):
        exps = []
        if not keys:
            return exps
        for k in keys:
            if k in self.memory:
                exps.append(self.memory[k])
        return exps

    def __len__(self):
        return len(self.memory)

    def merge_experiences(self, experiences: list, exp_keys: list):
        """
        Merge experiences into the agent's experience buffer.

        :param experiences: experiences to be merged.
        :param exp_keys: the keys of the shared experiences.
        :return:
        """
        self.log.debug('Merging experiences')

        initial_buffer_sz = len(self)

        # get reference data
        ref_data = self.sample(self.ref_data_sz)
        if ref_data:
            ref_data = np.array(ref_data)

            # convert exps to matrix
            experiences = np.array(experiences)

            # calculate cosine similarity with reference data
            sim = np.dot(experiences, ref_data.T) / (np.linalg.norm(experiences) * np.linalg.norm(ref_data))
            sim = np.max(sim, axis=1)

            # select indices which are less similar
            sel_idx = (sim < self.sim_threshold).nonzero()
            experiences = experiences[sel_idx].tolist()
            exp_keys = np.array(exp_keys)[sel_idx].tolist()

        # add selected exps to buffer
        for e_id, exp in zip(exp_keys, experiences):
            self.add_processed_exp(exp, e_id)

        if len(self) > initial_buffer_sz:
            self.log.debug(f'Experience merging completed: {initial_buffer_sz} -> {len(self)}')
