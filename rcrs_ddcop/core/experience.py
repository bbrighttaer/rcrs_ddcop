import datetime
import random
from collections import deque
from dataclasses import dataclass
from typing import List, Optional

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


class ExperienceBuffer:
    """Stores data for training models"""

    def __init__(self, capacity: int = 1000, lbl: str = ''):
        self.memory = CircularDict(maxlen=capacity)
        self.lbl = lbl

    def add(self, exp: List[List[Data]]) -> None:
        """Adds an experience to the buffer"""
        exp_id = self.lbl + '_' + str(round(datetime.datetime.now().timestamp()))
        self.memory[exp_id] = exp
        # save
        # if len(self) > 200:
        #     torch.save(self.memory, f'{self.lbl}.pt')

    def get_keys(self):
        return list(self.memory.keys())

    def sample(self, batch_size: int) -> List[List[Data]]:
        """
        Samples experiences from the buffer.

        :param batch_size: the number of samples to take from the buffer. If batch size is less than
                           the size of the buffer, a ValueError is thrown.
        :return: list of experiences.
        """
        if batch_size < len(self):
            return random.sample(self.memory.items(), batch_size)
        else:
            return random.sample(self.memory.items(), len(self))

    def select_exps_by_key(self, keys):
        exps = []
        for k in keys:
            exps.append(self.memory[k])
        return exps

    def __len__(self):
        return len(self.memory)


