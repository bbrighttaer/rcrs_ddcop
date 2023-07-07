import typing

import torch
from torch_geometric.data import Data, InMemoryDataset, Dataset


def data_to_dict(data: Data) -> dict:
    props = {
        'x': data.x.tolist(),
        'edge_index': data.edge_index.tolist(),
        'time_step': data.time_step,
        'entity_id': data.entity_id,
    }
    return props


def dict_to_data(data_dict: dict) -> Data:
    return Data(
        x=torch.tensor(data_dict['x']),
        edge_index=torch.tensor(data_dict['edge_index']),
        time_step=data_dict['time_step'],
        entity_id=data_dict['entity_id']
    )


class SimulationDataset(Dataset):

    def __init__(self, dataset: typing.List[Data]):
        super().__init__()
        self.data = dataset

    def len(self) -> int:
        return len(self.data)

    def get(self, idx: int):
        return self.data[idx]


