from rcrs_core.agents.agent import Agent

from rcrs_ddcop.algorithms.dcop.dpop import DPOP
from rcrs_ddcop.algorithms.graph.digca import DIGCA
from rcrs_ddcop.comm.pseudo_com import AgentPseudoComm


class BDIAgent(object):
    """
    Models a BDI-Agent properties
    """

    def __init__(self, agent: Agent):
        self.belief = agent.world_model
        self.log = agent.Log
        self.comm = AgentPseudoComm(agent, self.handle_message)
        self.graph = DIGCA(self)
        self.dcop = DPOP(self)
        self.agent_id = agent.agent_id
        self.domain = []

    @property
    def optimization_op(self):
        return 'max'

    @property
    def agents_in_range(self):
        return []

    @property
    def graph_traversing_order(self):
        return self.dcop.traversing_order

    @property
    def value(self):
        return None

    def objective(self, *args, **kwargs):
        """
        The desire is to optimize the objective functions in its neighborhood.
        :return:
        """
        return 0

    def deliberate(self):
        """
        Determines what the agent wants to do in the environment - its intention.
        :return:
        """
        ...

    def learn(self):
        ...

    def look_ahead(self):
        ...

    def handle_message(self, message):
        ...
