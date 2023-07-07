import random
import typing
from collections import deque
from dataclasses import dataclass


@dataclass
class Experience:
    state: typing.List
    action: typing.List
    utility: float
    next_state: typing.Optional[typing.List]

    def to_dict(self):
        return self.__dict__


class ExperienceBuffer:

    def __init__(self, capacity=1000):
        self.memory = deque([], maxlen=capacity)

    def add_ts_experience(self, time_step: int, exp: Experience):
        self.memory.append(exp)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


