import random
from collections import deque
from dataclasses import dataclass
from typing import List, Optional

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

    def __init__(self, capacity: int = 1000):
        self.memory = deque([], maxlen=capacity)

    def add(self, exp: List[List[Data]]) -> None:
        """Adds an experience to the buffer"""
        self.memory.append(exp)

    def sample(self, batch_size: int) -> List[List[Data]]:
        """
        Samples experiences from the buffer.

        :param batch_size: the number of samples to take from the buffer. If batch size is less than
                           the size of the buffer, a ValueError is thrown.
        :return: list of experiences.
        """
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


