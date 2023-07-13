import typing

import networkx as nx
import numpy as np
import torch
from rcrs_core.entities import standardEntityFactory
from rcrs_core.entities.area import Area
from rcrs_core.entities.building import Building
from rcrs_core.entities.human import Human
from rcrs_core.entities.road import Road
from rcrs_core.worldmodel.entityID import EntityID
from rcrs_core.worldmodel.worldmodel import WorldModel
from torch_geometric.data import Data, Dataset

from rcrs_ddcop.core.enums import Fieryness


def _get_unburnt_neighbors(world_model: WorldModel, building: Building):
    unburnt = []
    for n in building.get_neighbours():
        entity = world_model.get_entity(n)
        if isinstance(entity, Building):
            if entity.get_urn() == Building.urn and entity.get_fieryness() < Fieryness.BURNT_OUT:
                unburnt.append(entity)
    return unburnt


def world_to_state(world_model: WorldModel):
    # construct graph for the state
    world_graph = nx.Graph()

    # populate graph
    for entity in world_model.unindexedـentities.values():
        if isinstance(entity, Area):
            neighbors = entity.get_neighbours()
            for n in neighbors:
                n_entity = world_model.get_entity(n)
                if isinstance(n_entity, Area):
                    world_graph.add_edge(entity.get_id().get_value(), n.get_value())

        elif isinstance(entity, Human):
            pos_entity = world_model.get_entity(entity.position.get_value())
            if isinstance(pos_entity, Area):
                world_graph.add_edge(pos_entity.get_id().get_value(), entity.get_id().get_value())

    # construct node features
    node_features = []
    for node in world_graph.nodes:
        entity = world_model.get_entity(EntityID(node))
        if isinstance(entity, Building):
            node_features.append([
                                     entity.get_fieryness(),
                                     entity.get_temperature(),
                                     # entity.get_total_area(),
                                     entity.get_building_code(),
                                     len(_get_unburnt_neighbors(world_model, entity)),
                                 ] + [0.] * 3)
        elif isinstance(entity, Human):
            node_features.append([0.] * 4 + [
                entity.get_buriedness(),
                entity.get_damage(),
                entity.get_hp(),
            ])
        else:
            node_features.append([0.] * 7)

    adjacency_matrix = nx.adjacency_matrix(world_graph)
    rows, cols = np.nonzero(adjacency_matrix)
    edge_index_coo = torch.tensor(np.array(list(zip(rows, cols))).reshape(2, -1), dtype=torch.long)
    node_feat_arr = torch.tensor(node_features, dtype=torch.float)
    data = Data(
        x=node_feat_arr,
        edge_index=edge_index_coo,
        nodes_order=list(world_graph.nodes.keys()),
        node_urns=[world_model.get_entity(EntityID(n)).get_urn().value for n in world_graph.nodes]
    )

    return data


def state_to_world(data: Data):
    world_model = WorldModel()

    for feat, node_id, node_urn in zip(data.x, data.nodes_order, data.node_urns):
        entity = standardEntityFactory.StandardEntityFactory.make_entity(
            urn=node_urn,
            id=node_id
        )
        if isinstance(entity, Building):
            entity.set_fieryness(feat[0].item()),
            entity.set_temperature(feat[1].item()),
            entity.set_building_code(feat[2].item())
        elif isinstance(entity, Human):
            entity.set_buriedness(feat[4].item()),
            entity.set_damage(feat[5].item()),
            entity.set_hp(feat[6].item()),
        world_model.add_entity(entity)

    return world_model


def state_to_dict(data: Data):
    return {
        'x': data.x.tolist(),
        'edge_index': data.edge_index.tolist(),
        'nodes_order': data.nodes_order,
        'node_urns': data.node_urns,
    }


def dict_to_state(data: dict):
    return Data(
        x=torch.tensor(data['x'], dtype=torch.float),
        edge_index=torch.tensor(data['edge_index'], dtype=torch.long),
        nodes_order=data['nodes_order'],
        node_urns=data['node_urns'],
    )


class SimulationDataset(Dataset):

    def __init__(self, dataset: typing.List[Data]):
        super().__init__()
        self.data = dataset

    def len(self) -> int:
        return len(self.data)

    def get(self, idx: int):
        return self.data[idx]
