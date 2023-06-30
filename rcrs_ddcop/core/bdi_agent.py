import datetime
import threading

import numpy as np
from rcrs_core.agents.agent import Agent

from rcrs_ddcop.algorithms.dcop.dpop import DPOP
from rcrs_ddcop.algorithms.dcop.lsla import LSLA
from rcrs_ddcop.algorithms.graph.digca import DIGCA
from rcrs_ddcop.comm import messaging
from rcrs_ddcop.comm.pseudo_com import AgentPseudoComm
from rcrs_ddcop.core.experience import Experience, ExperienceBuffer


class BDIAgent(object):
    """
    Models a BDI-Agent properties
    """

    def __init__(self, agent: Agent):
        self.latest_event_timestamp = None
        self.belief = agent.world_model
        self.log = agent.Log
        self.agent_id = agent.agent_id.get_value()
        self._domain = []
        self._state = None
        self._agents_in_comm_range = []
        self._new_agents = []
        self._value = None
        self._previous_value = None
        self._terminate = False
        self._neighbor_domains = {}
        self._neighbor_previous_values = {}
        self._binary_constraint = agent.binary_constraint
        self._experience_buffer: ExperienceBuffer = agent.experience_buffer

        # manages when control is returned to agent entity
        self._value_selection_evt = threading.Event()

        # create instances of main components
        self.comm = AgentPseudoComm(agent, self.handle_message)
        self.graph = DIGCA(self)
        self.dcop = LSLA(self, self.on_value_selected)

    @property
    def domain(self):
        return self._domain

    @domain.setter
    def domain(self, v):
        self._domain = v

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, s):
        self._state = s

    @property
    def agents_in_comm_range(self):
        return self._agents_in_comm_range

    @agents_in_comm_range.setter
    def agents_in_comm_range(self, v):
        self._agents_in_comm_range = v

    @property
    def new_agents(self):
        return self._new_agents

    @property
    def optimization_op(self):
        return 'max'

    @property
    def graph_traversing_order(self):
        return self.dcop.traversing_order

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, v):
        self._value = v

    @property
    def neighbor_domains(self):
        return self._neighbor_domains

    @property
    def previous_value(self):
        return self._previous_value

    def add_neighbor_domain(self, k, v):
        self._neighbor_domains[k] = v

    def add_neighbor_previous_value(self, k, v):
        self._neighbor_previous_values[k] = v

    def remove_neighbor_domain(self, k):
        if k in self._neighbor_domains:
            del self._neighbor_domains[k]

    def on_value_selected(self, value, *args, **kwargs):
        self._value = value
        self._value_selection_evt.set()

    def execute_dcop(self):
        self.dcop.execute_dcop()

    def clear_current_value(self):
        self._previous_value = self._value
        self._value = None

    def belief_revision_function(self):
        ...

    def objective(self, agent_vals: dict):
        """
        The desire is to optimize the objective functions in its neighborhood.
        :return:
        """
        score = self._binary_constraint(agent_vals)
        return score

    def share_information(self, **kwargs):
        """
        Shares information with neighbors
        """
        self.comm.share_information_with_neighbors(
            neighbor_ids=self.graph.neighbors,
            data={
                'agent_id': self.agent_id,
                'domain': self._domain,
                'previous_value': self._previous_value,
                **kwargs,
            },
        )

    def deliberate(self, state):
        """
        Determines what the agent wants to do in the environment - its intention.
        :return:
        """
        # set time step or cycle properties
        self.latest_event_timestamp = datetime.datetime.now().timestamp()
        self._new_agents = set(self.agents_in_comm_range) - set(self.graph.neighbors)
        self._value_selection_evt.clear()

        self._utilities = np.zeros((len(self.domain), 1))

        # clear currently assigned value
        self.clear_current_value()

        # remove agents that are out-of-range
        agents_to_remove = set(self.graph.neighbors) - set(self.agents_in_comm_range)
        if agents_to_remove:
            for _agent in agents_to_remove:
                self.graph.remove_agent(_agent)

        self.log.info(
            f'parent={self.graph.parent}, '
            f'children={self.graph.children}, '
            f'agents-in-range={self.agents_in_comm_range}'
        )

        # clear time step buffers
        self.dcop.on_time_step_changed()
        self.graph.on_time_step_changed()

        # if no neighborhood change
        if not self.graph.has_potential_neighbor():
            self.comm.threadsafe_execution(self.graph.start_dcop)

        # wait for value section or timeout
        self._value_selection_evt.wait()
        return self._value, self.dcop.cost

    def learn(self):
        ...

    def look_ahead(self):
        ...

    def handle_message(self, message):
        # reject outdated messages (every message has a timestamp)
        if self.latest_event_timestamp and message['timestamp'] < self.latest_event_timestamp:
            return

        match message['type']:
            # DIGCA message handling
            case messaging.DIGCAMsgTypes.ANNOUNCE:
                self.graph.receive_announce(message)

            case messaging.DIGCAMsgTypes.ANNOUNCE_RESPONSE:
                self.graph.receive_announce_response(message)

            case messaging.DIGCAMsgTypes.ANNOUNCE_RESPONSE_IGNORED:
                self.graph.receive_announce_response_ignored(message)

            case messaging.DIGCAMsgTypes.ADD_ME:
                self.graph.receive_add_me(message)

            case messaging.DIGCAMsgTypes.CHILD_ADDED:
                self.graph.receive_child_added(message)

            case messaging.DIGCAMsgTypes.PARENT_ASSIGNED:
                self.graph.receive_parent_assigned(message)

            case messaging.DIGCAMsgTypes.ALREADY_ACTIVE:
                self.graph.receive_already_active(message)

            # DPOP message handling
            case messaging.DPOPMsgTypes.VALUE_MESSAGE:
                self.dcop.receive_value_message(message)

            case messaging.DPOPMsgTypes.UTIL_MESSAGE:
                self.dcop.receive_util_message(message)

            case messaging.DPOPMsgTypes.REQUEST_UTIL_MESSAGE:
                self.dcop.receive_util_message_request(message)

            # Other agent communication
            case messaging.AgentMsgTypes.SHARED_INFO:
                self.receive_shared_info(message)

            # LSLA message handling
            case messaging.LSLAMsgTypes.LSLA_INQUIRY_MESSAGE:
                self.dcop.receive_inquiry_message(message)

            case messaging.LSLAMsgTypes.LSLA_UTIL_MESSAGE:
                self.dcop.receive_util_message(message)

            case _:
                self.log.info(f'Could not handle received payload: {message}')

    def receive_shared_info(self, message: dict):
        self.log.info(f'Received shared message')
        data = message['payload']
        sender = data['agent_id']
        shared_exp = data.get('shared_exp')
        self.add_neighbor_domain(sender, data['domain'])
        self.add_neighbor_previous_value(sender, data['previous_value'])

        if shared_exp:
            time_step = shared_exp['time_step']
            exp = Experience(
                state=shared_exp['exp']['state'],
                action=shared_exp['exp']['action'],
                utility=shared_exp['exp']['utility'],
                next_state=shared_exp['exp']['next_state'],
            )
            self._experience_buffer.add_ts_experience(
                time_step=time_step,
                exp=exp,
            )

    def __call__(self, *args, **kwargs):
        self.log.info(f'Initializing agent {self.agent_id}')
        while not self._terminate:
            self.comm.listen_to_network()
            self.graph.connect()
            self.dcop.resolve_value()

