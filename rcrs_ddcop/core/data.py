import math
from typing import Iterable

import networkx as nx
import numpy as np
import pandas as pd
import smogn
import torch
from rcrs_core.entities import standardEntityFactory
from rcrs_core.entities.area import Area
from rcrs_core.entities.building import Building
from rcrs_core.entities.human import Human
from rcrs_core.worldmodel.entityID import EntityID
from rcrs_core.worldmodel.worldmodel import WorldModel
from torch_geometric.data import Data
import ImbalancedLearningRegression as iblr

from rcrs_ddcop.core.enums import Fieryness


def _get_unburnt_neighbors(world_model: WorldModel, building: Building) -> list:
    """Gets the list of unburnt buildings close to the given building"""
    unburnt = []
    for n in building.get_neighbours():
        entity = world_model.get_entity(n)
        if isinstance(entity, Building):
            if entity.get_urn() == Building.urn and entity.get_fieryness() < Fieryness.BURNT_OUT:
                unburnt.append(entity)
    return unburnt


def world_to_state(world_model: WorldModel, entity_ids: Iterable[int] = None, edge_index: torch.Tensor = None) -> Data:
    """
    Extracts properties from the given world to construct a Pytorch Geometric (PyG) data instance.

    :param world_model: The world model or belief to lookup entities
    :param entity_ids: list of entities for constructing the state.
    :param edge_index: precomputed edge index. Must be supplied if `entity_ids` is passed.
    :return: parsed world to `Data` object.
    """
    # construct graph for the state
    world_graph = nx.Graph()
    edge_index_coo = None

    if not entity_ids:
        # populate graph
        state_entities = [EntityID(e) for e in entity_ids] if entity_ids else world_model.unindexedـentities
        for entity_id in state_entities:
            entity = world_model.get_entity(entity_id)
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

        # construct edge index
        adjacency_matrix = nx.adjacency_matrix(world_graph)
        rows, cols = np.nonzero(adjacency_matrix)
        edge_index_coo = torch.tensor(np.array(list(zip(rows, cols))).reshape(2, -1), dtype=torch.long)

    # construct node features
    node_features = []
    for node in entity_ids if entity_ids else world_graph.nodes:
        entity = world_model.get_entity(EntityID(node))
        if isinstance(entity, Building):
            node_features.append([
                entity.get_fieryness(),
                entity.get_temperature(),
                entity.get_brokenness(),
                entity.get_building_code(),
                get_building_fire_index(entity, world_model),
                0,
            ])
            # ] + [0.] * 3)
        # elif isinstance(entity, Human):
        #     node_features.append([0.] * 5 + [
        #         entity.get_buriedness(),
        #         entity.get_damage(),
        #         entity.get_hp(),
        #     ])
        else:
            node_features.append([0.] * 6)

    node_feat_arr = torch.tensor(node_features, dtype=torch.float)
    data = Data(
        x=node_feat_arr,
        edge_index=edge_index if edge_index is not None else edge_index_coo,
        nodes_order=list(world_graph.nodes.keys()),
        node_urns=[world_model.get_entity(EntityID(n)).get_urn().value for n in world_graph.nodes]
    )

    return data


def state_to_world(data: Data) -> WorldModel:
    """Converts a PyG data object to a World model"""
    world_model = WorldModel()

    for feat, node_id, node_urn in zip(data.x, data.nodes_order, data.node_urns):
        entity = standardEntityFactory.StandardEntityFactory.make_entity(
            urn=node_urn,
            id=node_id
        )
        if isinstance(entity, Building):
            entity.set_fieryness(int(feat[0].item()))
            entity.set_temperature(int(feat[1].item()))
            entity.set_brokenness(int(feat[2].item()))
            entity.set_building_code(int(feat[3].item()))
        # elif isinstance(entity, Human):
        #     entity.set_buriedness(round(feat[5].item()))
        #     entity.set_damage(round(feat[6].item()))
        #     entity.set_hp(round(feat[7].item()))
        world_model.add_entity(entity)

    return world_model


def state_to_dict(data: Data) -> dict:
    """Converts a PyG data object to python dictionary"""
    return {
        'x': data.x.tolist(),
        'edge_index': data.edge_index.tolist(),
        'nodes_order': data.nodes_order,
        'node_urns': data.node_urns,
    }


def dict_to_state(data: dict) -> Data:
    """Reverses a PyG data object to dictionary conversion"""
    return Data(
        x=torch.tensor(data['x'], dtype=torch.float),
        edge_index=torch.tensor(data['edge_index'], dtype=torch.long),
        nodes_order=data['nodes_order'],
        node_urns=data['node_urns'],
    )


def process_data(raw_data: list[list[Data]], transform=None) -> list[Data]:
    data = []
    transform_data = []
    for record in raw_data:
        state = record[0]
        s_prime = record[1]

        # remove uninformative rows
        idx = state.x[:, 0] != s_prime.x[:, 0]
        state.x = state.x[idx]
        s_prime.x = s_prime.x[idx]
        if len(state.x) == 0 or len(s_prime.x) == 0:
            continue

        # identify change in fieriness
        diff = torch.clip(state.x[:, 0] - s_prime.x[:, 0], 0, 1).view(-1, 1)
        state.x = torch.concat([state.x[:, :-1], diff], dim=1)

        # gather data for computing normalization statistics
        transform_data.extend([state.x.numpy(), s_prime.x.numpy()])

        # create data and add it to the set
        d_instance = Data(
            x=state.x,
            y=s_prime.x,
            edge_index=state.edge_index,
            nodes_order=state.nodes_order,
            node_urns=state.node_urns,
        )
        data.append(d_instance)

    # compute data normalization stats
    if transform_data:
        transform.fit(np.concatenate(transform_data))
    return data


def get_building_fire_index(building: Building, world_model: WorldModel):
    neighbor_temps = []
    for neighbor in building.get_neighbours():
        entity = world_model.get_entity(neighbor)
        if isinstance(entity, Building):
            neighbor_temps.append(entity.get_temperature())
    return max(neighbor_temps) if neighbor_temps else 0.


def correct_skewed_data(X, Y, columns, target_col):
    data = np.concatenate([X, Y], axis=1)

    # see https://github.com/nickkunz/smogn/blob/master/examples/smogn_example_3_adv.ipynb
    rg_mtrx = [
        [0, 0, 0],  ## under-sample
        [1, 1, 0],  ## over-sample
        [2, 1, 0],  ## over-sample
        [3, 1, 0],  ## over-sample
        [4, 1, 0],  ## under-sample
        [5, 1, 0],  ## under-sample
        [6, 1, 0],  ## under-sample
        [7, 0, 0],  ## under-sample
        [8, 0, 0],  ## under-sample
    ]
    # data_bal = smogn.smoter(
    #     data=pd.DataFrame(data, columns=columns),
    #     y=target_col,
    #     rel_thres=0.1,
    #     rel_method='manual',
    #     rel_ctrl_pts_rg=rg_mtrx,
    # )
    data_bal = iblr.gn(
        data=pd.DataFrame(data, columns=columns),
        y='fieryness_x',
        rel_thres=0.5,
        rel_method='manual',
        rel_ctrl_pts_rg=rg_mtrx,
    )
    data_sampled = data_bal.to_numpy()
    X_ = data_sampled[:, :X.shape[-1]]
    Y_ = data_sampled[:, X.shape[-1]:]
    return X_, Y_

