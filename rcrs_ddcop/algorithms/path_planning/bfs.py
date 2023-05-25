from collections import defaultdict
from typing import List

from rcrs_core.entities.area import Area
from rcrs_core.entities.building import Building
from rcrs_core.entities.refuge import Refuge
from rcrs_core.worldmodel.entityID import EntityID
from rcrs_core.worldmodel.worldmodel import WorldModel


class BFSSearch:
    """
    Implements Breadth-first search path planning
    """

    def __init__(self, world_model: WorldModel):
        self.building_set = set()
        self.graph = defaultdict(set)

        # construct graph from entities in the world
        for entity in world_model.get_entities():
            if isinstance(entity, Area):
                area_neighbors = entity.get_neighbours()
                self.graph[entity.get_id()].update(area_neighbors)

                if isinstance(entity, Building):
                    self.building_set.add(entity)

    def breadth_first_search(self, start: EntityID, goals: List[EntityID]):
        if not goals:
            return None

        open_list = []
        ancestors = {}

        current_entity = None
        found = False
        open_list.append(start)
        ancestors[start] = start

        while not found and open_list:
            current_entity = open_list.pop(0)
            if self.is_goal(current_entity, goals):
                found = True
                break

            neighbors = self.graph[current_entity]
            for neighbor in neighbors:
                if self.is_goal(neighbor, goals):
                    ancestors[neighbor] = current_entity
                    current_entity = neighbor
                    found = True
                    break

                # ignore already visited neighbors
                if neighbor not in ancestors:
                    open_list.append(neighbor)
                    ancestors[neighbor] = current_entity

        if not found:
            return None
        else:
            path = [current_entity.get_value()]
            while current_entity != start:
                current_entity = ancestors[current_entity]
                if current_entity is None:
                    raise RuntimeError('Found a node with no ancestor! Something is broken.')
                path.append(current_entity.get_value())
            return list(reversed(path))

    def breadth_first_search_for_civilian(self, start: EntityID, goals: List[EntityID]):
        open_list = []
        ancestors = {}

        current_entity = None
        found = False
        open_list.append(start)
        ancestors[start] = start

        while not found and open_list:
            current_entity = open_list.pop(0)
            if self.is_goal(current_entity, goals):
                found = True
                break

            neighbors = self.graph[current_entity]
            if neighbors:
                neighbors_iter = iter(neighbors)

                while True:
                    try:
                        neighbor = next(neighbors_iter)
                        while current_entity not in self.building_set and neighbor in self.building_set:

                            # ignore already visited neighbors
                            while neighbor in ancestors:
                                # check if this neighbor is the goal
                                if self.is_goal(neighbor, goals):
                                    ancestors[neighbor] = current_entity
                                    current_entity = neighbor
                                    found = True
                                    break

                                # visit next neighbor
                                neighbor = next(neighbors_iter)

                            if found:
                                break

                        if found:
                            break

                        open_list.append(neighbor)
                        ancestors[neighbor] = current_entity

                    except StopIteration:
                        break

        if not found:
            return None
        else:
            path = [current_entity]
            while current_entity != start:
                current_entity = ancestors[current_entity]
                if current_entity is None:
                    raise RuntimeError('Found a node with no ancestor! Something is broken.')
                path.append(current_entity)
            return list(reversed(path))

    def is_goal(self, e: EntityID, test: List[EntityID]):
        return e in test

