from collections import deque, defaultdict
from typing import Callable, List

from rcrs_ddcop.core.enums import DynamicGraphCallback


def get_agent_order(agent_id):
    return agent_id  # int(agent_id.replace('a', ''))


class Unsupported(Exception):

    def __init__(self):
        super().__init__('Operation is not implemented')


class DynaGraph:
    """
    Base class for dynamic graph algorithms
    """

    def __init__(self, agent):
        self.agent = agent
        self.comm = agent.comm
        self.parent = None
        self.children = deque()
        self.pseudo_children = deque()
        self.pseudo_parents = deque()
        self.children_history = {}
        self.log = self.agent.log
        self.exec_started = False
        self._callback_register = defaultdict(list)

    def is_neighbor(self, agent_id):
        return self.parent == agent_id or agent_id in self.children

    def is_child(self, agent_id):
        return agent_id in self.children

    def is_parent(self, agent_id):
        return self.parent == agent_id

    @property
    def num_of_neighbors(self):
        return len(self.children) + (1 if self.parent else 0)

    @property
    def neighbors(self):
        """only parent and children"""
        neighbors = []
        if self.children:
            neighbors.extend(self.children)
        if self.parent:
            neighbors.append(self.parent)
        return neighbors

    @property
    def all_neighbors(self):
        """considers parent, children, pseudo-parents, and pseudo-children"""
        return self.get_connected_agents()

    @property
    def all_parents(self):
        """parent and pseudo-parents"""
        parents = list(self.pseudo_parents)
        if self.parent:
            parents.append(self.parent)
        return parents

    @property
    def all_children(self):
        """true and pseudo-children"""
        return list(self.children) + list(self.pseudo_children)

    def start_dcop(self, timeout=False):
        fully_connected = len(self.all_neighbors) == len(self.agent.agents_in_comm_range)
        if not self.agent.value and (timeout or fully_connected):
            self.log.info(f'Starting DCOP from DIGCA')
            self.agent.execute_dcop()
            self.exec_started = True
        else:
            self.log.info(f'DCOP not started, waiting for {self.agent.new_agents}')

    def has_potential_parent(self):
        raise Unsupported()

    def has_potential_child(self):
        raise Unsupported()

    def has_potential_neighbor(self):
        raise Unsupported()

    def report_connection(self, parent, child, constraint):
        # self.address_table.basic_publish(exchange=messaging.COMM_EXCHANGE,
        #                            routing_key=f'{messaging.MONITORING_CHANNEL}',
        #                            body=messaging.create_agent_connection_message({
        #                                'agent_id': self.agent.agent_id,
        #                                'child': child,
        #                                'parent': parent,
        #                                'constraint': str(constraint),
        #                            }))
        ...

    def report_agent_disconnection(self, agent):
        # inform dashboard about disconnection
        # self.address_table.basic_publish(
        #     exchange=messaging.COMM_EXCHANGE,
        #     routing_key=f'{messaging.MONITORING_CHANNEL}',
        #     body=messaging.create_agent_disconnection_message({
        #         'agent_id': self.agent.agent_id,
        #         'node1': self.agent.agent_id,
        #         'node2': agent,
        #     })
        # )

        # update current graph
        # self.address_table.basic_publish(
        #     exchange=messaging.COMM_EXCHANGE,
        #     routing_key=f'{messaging.SIM_ENV_CHANNEL}',
        #     body=messaging.create_remove_graph_edge_message({
        #         'agent_id': self.agent.agent_id,
        #         'from': agent,
        #         'to': self.agent.agent_id,
        #     })
        # )
        ...

    def get_connected_agents(self):
        """considers parent, children, pseudo-parents, and pseudo-children"""
        cons = self.children + self.pseudo_children + self.pseudo_parents
        if self.parent:
            cons += [self.parent]
        return cons

    def register_callback(self, cb_type: DynamicGraphCallback, cb_func: Callable):
        self._callback_register[cb_type].append(cb_func)

    def remove_callback(self, cb_type: DynamicGraphCallback, cb_func: Callable):
        if cb_func in self._callback_register[cb_type]:
            self._callback_register[cb_type].remove(cb_func)

    def fire_callbacks(self, cb_types: List[DynamicGraphCallback], **kwargs):
        for cb_type in cb_types:
            for cb in self._callback_register[cb_type]:
                cb(**kwargs)
