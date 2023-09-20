from collections import deque


def get_agent_order(agent_id):
    return agent_id  # int(agent_id.replace('a', ''))


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

    def has_no_neighbors(self):
        return not self.parent and not self.children and not self.pseudo_parents and self.pseudo_children

    def is_neighbor(self, agent_id):
        return self.parent == agent_id or agent_id in self.children

    def is_child(self, agent_id):
        return agent_id in self.children

    def is_parent(self, agent_id):
        return self.parent == agent_id

    @property
    def num_of_neighbors(self):
        return len(self.children) + 1 if self.parent else 0

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
        neighbors = self.neighbors + list(self.pseudo_children) + list(self.pseudo_parents)
        return neighbors

    def start_dcop(self):
        self.log.debug(f'Starting DCOP...')
        self.agent.execute_dcop()
        self.exec_started = True

    def report_connection(self, parent, child, constraint):
        # self.channel.basic_publish(exchange=messaging.COMM_EXCHANGE,
        #                            routing_key=f'{messaging.MONITORING_CHANNEL}',
        #                            body=messaging.create_agent_connection_message({
        #                                'agent_id': self.agent.agent_id,
        #                                'child': child,
        #                                'parent': parent,
        #                                'constraint': str(constraint),
        #                            }))
        ...

    def has_potential_parent(self):
        ...

    def has_potential_child(self):
        ...

    def has_potential_neighbor(self):
        ...

    def report_agent_disconnection(self, agent):
        # inform dashboard about disconnection
        # self.channel.basic_publish(
        #     exchange=messaging.COMM_EXCHANGE,
        #     routing_key=f'{messaging.MONITORING_CHANNEL}',
        #     body=messaging.create_agent_disconnection_message({
        #         'agent_id': self.agent.agent_id,
        #         'node1': self.agent.agent_id,
        #         'node2': agent,
        #     })
        # )

        # update current graph
        # self.channel.basic_publish(
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
        cons = self.children + self.pseudo_children + self.pseudo_parents
        if self.parent:
            cons += [self.parent]
        return cons
