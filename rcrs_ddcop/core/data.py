import math
from collections import defaultdict
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
from rcrs_ddcop.utils.common_funcs import euclidean_distance


def _get_unburnt_neighbors(world_model: WorldModel, building: Building) -> list:
    """Gets the list of unburnt buildings close to the given building"""
    unburnt = []
    for n in building.get_neighbours():
        entity = world_model.get_entity(n)
        if isinstance(entity, Building):
            if entity.get_urn() == Building.urn and entity.get_fieryness() < Fieryness.COMPLETELY_BURNT:
                unburnt.append(entity)
    return unburnt


def world_to_state(world_model: WorldModel, entity_ids: Iterable[int] = None, edge_index: torch.Tensor = None) -> Data:
    """
    Extracts properties from the given world to construct a Pytorch Geometric (PyG) train_data instance.

    :param world_model: The world model or belief to lookup entities
    :param entity_ids: list of entities for constructing the state.
    :param edge_index: precomputed edge index. Must be supplied if `entity_ids` is passed.
    :return: parsed world to `Data` object.
    """
    state_entities = [EntityID(e) for e in entity_ids] if entity_ids else world_model.unindexedـentities
    buildings = []
    for entity_id in state_entities:
        entity = world_model.get_entity(entity_id)

        # find buildings
        if isinstance(entity, Building):
            buildings.append(entity)

    # get building neighbors based on distance metric
    building_to_neighbors = defaultdict(list)
    for building in buildings:
        for b2 in buildings:
            dist = euclidean_distance(building.get_x(), building.get_y(), b2.get_x(), b2.get_y())
            if b2 != building and dist < 30000:
                building_to_neighbors[building.get_id()].append(b2)

    # compute fire index for each building
    b_fire_idx = {
        b: max([nb.get_temperature() for nb in building_to_neighbors[b]]) for b in building_to_neighbors
    }

    # construct node features
    node_features = []
    nodes_order = []
    node_urns = []
    for e_id in state_entities:
        entity = world_model.get_entity(e_id)
        nodes_order.append(e_id.get_value())
        node_urns.append(entity.get_urn().value)
        if isinstance(entity, Building):
            node_features.append([
                entity.get_temperature(),
                entity.get_fieryness(),
                entity.get_brokenness(),
                entity.get_building_code(),
                b_fire_idx[entity.get_id()] if entity.get_id() in b_fire_idx else entity.get_temperature(),
            ])
            # ] + [0.] * 3)
        # elif isinstance(entity, Human):
        #     node_features.append([0.] * 5 + [
        #         entity.get_buriedness(),
        #         entity.get_damage(),
        #         entity.get_hp(),
        #     ])
        else:
            node_features.append([0.] * 5)

    node_feat_arr = torch.tensor(node_features, dtype=torch.float)
    data = Data(
        x=node_feat_arr,
        nodes_order=nodes_order,
        node_urns=node_urns,
    )

    return data


def state_to_world(data: Data) -> WorldModel:
    """Converts a PyG train_data object to a World model"""
    world_model = WorldModel()

    for feat, node_id, node_urn in zip(data.x, data.nodes_order, data.node_urns):
        entity = standardEntityFactory.StandardEntityFactory.make_entity(
            urn=node_urn,
            id=node_id
        )
        if isinstance(entity, Building):
            entity.set_temperature(int(feat[0].item()))
            entity.set_fieryness(int(feat[1].item()))
            entity.set_brokenness(int(feat[2].item()))
            entity.set_building_code(int(feat[3].item()))
        # elif isinstance(entity, Human):
        #     entity.set_buriedness(round(feat[5].item()))
        #     entity.set_damage(round(feat[6].item()))
        #     entity.set_hp(round(feat[7].item()))
        world_model.add_entity(entity)

    return world_model


def state_to_dict(data: Data) -> dict:
    """Converts a PyG train_data object to python dictionary"""
    return {
        'val_data': data.x.tolist(),
        'nodes_order': data.nodes_order,
        'node_urns': data.node_urns,
    }


def dict_to_state(data: dict) -> Data:
    """Reverses a PyG train_data object to dictionary conversion"""
    return Data(
        x=torch.tensor(data['val_data'], dtype=torch.float),
        nodes_order=data['nodes_order'],
        node_urns=data['node_urns'],
    )


def get_building_fire_index(building: Building, world_model: WorldModel):
    neighbor_temps = []
    for neighbor in building.get_neighbours():
        entity = world_model.get_entity(neighbor)
        if isinstance(entity, Building):
            neighbor_temps.append(entity.get_temperature())
    val = max(neighbor_temps) if neighbor_temps else 0.
    print(f'building index: {val}')
    return val


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
    #     train_data=pd.DataFrame(train_data, columns=columns),
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

