import datetime
import functools
import threading
from collections import deque

from rcrs_core.agents.agent import Agent

from rcrs_ddcop.algorithms.dcop.la_cocoa import LA_CoCoA
from rcrs_ddcop.algorithms.graph.digca import DIGCA
from rcrs_ddcop.comm import messaging
from rcrs_ddcop.comm.pseudo_com import AgentPseudoComm
from rcrs_ddcop.core.data import dict_to_state
from rcrs_ddcop.core.enums import InfoSharingType
from rcrs_ddcop.core.experience import ExperienceBuffer


class BDIAgent(object):
    """
    Models a BDI-Agent properties
    """

    def __init__(self, agent: Agent):
        self._rcrs_agent = agent
        self.agent_id = agent.agent_id.get_value()
        self.label = f'{agent.name}_{self.agent_id}'
        self.latest_event_timestamp = None
        self.belief = agent.world_model
        self.log = agent.Log
        self._domain = []
        self._state = None
        self._agents_in_comm_range = []
        self._new_agents = []
        self._value = None
        self._previous_value = None
        self._terminate = False
        self._neighbor_domains = {}
        self._neighbor_previous_values = {}
        self.experience_buffer = ExperienceBuffer(lbl=self.label)
        self._timeout = 1.5

        self._decision_timeout_count = 0

        # agent-type constraint functions
        self.agent_type_neighbor_constraint = agent.neighbor_constraint
        self.unary_constraint = agent.unary_constraint

        # manages when control is returned to agent entity
        self._value_selection_evt = threading.Event()

        # paused messages queue
        self.paused_messages = deque()

        # create instances of main components
        self.comm = AgentPseudoComm(self)
        self.graph = DIGCA(self, timeout=self._timeout, max_num_of_neighbors=3)
        self.dcop = LA_CoCoA(self, self.on_value_selected, label=self.label)

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

    @property
    def time_step(self):
        return self._rcrs_agent.current_time_step

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

    def neighbor_constraint(self, *args, **kwargs):
        """
        The desire is to optimize the objective functions in its neighborhood.
        :return:
        """
        return self.agent_type_neighbor_constraint(*args, **kwargs)

    def share_information(self, **kwargs):
        """
        Shares information with neighbors
        """
        self.comm.share_information_with_agents(
            agent_ids=self.graph.neighbors,
            data={
                'agent_id': self.agent_id,
                'domain': self._domain,
                'previous_value': self._previous_value,
                **kwargs,
            },
            sharing_type=InfoSharingType.STATE_SHARING,
        )

    def share_buried_entities_information(self, agent_ids: list, data):
        """
        Shares information of buried entities with the given agents
        """
        self.comm.share_information_with_agents(
            agent_ids=agent_ids,
            data={
                'agent_id': self.agent_id,
                'buried_info': data,
            },
            sharing_type=InfoSharingType.BURIED_HUMAN_SHARING,
        )

    def deliberate(self, time_step):
        """
        Determines what the agent wants to do in the environment - its intention.
        :return:
        """
        self.log.debug('started deliberation...')
        # set time step or cycle properties
        self.latest_event_timestamp = datetime.datetime.now().timestamp()
        self._new_agents = set(self.agents_in_comm_range) - set(self.graph.neighbors)
        self._value_selection_evt.clear()

        # clear time step buffers
        self.dcop.on_time_step_changed()
        self.graph.on_time_step_changed()

        # clear currently assigned value
        self.clear_current_value()

        if len(self.graph.children) > 0 and len(self.agents_in_comm_range) == 0:
            self.log.warning(f'Inconsistent children list: '
                             f'{self.graph.children}, neighbors in range = {self.agents_in_comm_range}')

        self.log.info(
            f'parent={self.graph.parent}, '
            f'children={self.graph.children}, '
            f'agents-in-range={self.agents_in_comm_range}, '
            f'domain={self.domain}'
        )

        self.process_paused_msgs()

        # if no neighborhood change
        if not self.graph.has_potential_neighbor():
            self.comm.threadsafe_execution(self.graph.start_dcop)
        else:
            self.log.info('waiting for potential connections')

        # wait for value section or timeout
        if not self._value_selection_evt.wait(timeout=self._timeout):
            self._decision_timeout_count += 1
            self.dcop.record_agent_metric('decision timeout', time_step, self._decision_timeout_count)
            self.log.warning('Agent decision timeout')

        # if no value is selected after timeout, select a random value
        if self._value is None:
            self.comm.threadsafe_execution(self.dcop.select_random_value)
            self._value_selection_evt.wait()

        self.log.debug('finished deliberation...')
        return self._value, self.dcop.cost

    def remove_unreachable_neighbors(self):
        # remove agents that are out-of-range
        agents_to_remove = set(self.graph.neighbors) - set(self.agents_in_comm_range)
        if agents_to_remove:
            self.log.debug(f'Removing {agents_to_remove}')
            for _agent in agents_to_remove:
                self.graph.remove_agent(_agent)
            self.log.debug(f'children={self.graph.children}, parent={self.graph.parent}')

    def process_paused_msgs(self):
        num_msgs = len(self.paused_messages)
        mgs_queue = self.paused_messages
        if num_msgs > 0:
            self.log.info(f'Processing {num_msgs} paused messages')
            self.paused_messages = deque()
        for msg in mgs_queue:
            self.comm.threadsafe_execution(functools.partial(self.handle_message, msg, True))
        self.paused_messages.clear()

    def handle_message(self, message, is_delayed=False):
        # reject outdated messages (every message has a timestamp)
        if not is_delayed and self.latest_event_timestamp and message['timestamp'] < self.latest_event_timestamp:
            self.log.warning(f'Cannot handle past message: {message}')
            return

        # store messages of future time step for processing when the agent is ready for this future time step
        if message['time_step'] > self.time_step:
            self.paused_messages.append(message)
            self.log.debug(f'Added {message["type"]} message from {message["payload"]["agent_id"]} to paused queue')
        else:
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

                # CoCoA message handling
                case messaging.CoCoAMsgTypes.INQUIRY_MESSAGE:
                    self.dcop.receive_inquiry_message(message)

                case messaging.CoCoAMsgTypes.COST_MESSAGE:
                    self.dcop.receive_cost_message(message)

                case messaging.CoCoAMsgTypes.UPDATE_STATE_MESSAGE:
                    self.dcop.receive_update_state_message(message)

                case messaging.CoCoAMsgTypes.EXECUTION_REQUEST:
                    self.dcop.receive_execution_request_message(message)

                # Information sharing
                case messaging.AgentMsgTypes.SHARED_INFO:
                    self.receive_shared_info(message, InfoSharingType.STATE_SHARING)

                case messaging.AgentMsgTypes.SHARED_BURIED_ENTITIES:
                    self.receive_shared_info(message, InfoSharingType.BURIED_HUMAN_SHARING)

                # LSLA message handling
                case messaging.LSLAMsgTypes.LSLA_INQUIRY_MESSAGE:
                    self.dcop.receive_inquiry_message(message)

                case messaging.LSLAMsgTypes.LSLA_UTIL_MESSAGE:
                    self.dcop.receive_util_message(message)

                # General
                case messaging.AgentMsgTypes.BUSY:
                    self.handle_busy_agent(message)

                case _:
                    self.log.info(f'Could not handle received payload: {message}')

    def receive_shared_info(self, message: dict, message_type: InfoSharingType):
        data = message['payload']
        sender = data['agent_id']
        self.log.info(f'Received shared message from {sender}')

        if message_type == InfoSharingType.STATE_SHARING:
            shared_exp = data.get('exp')
            self.add_neighbor_domain(sender, data['domain'])
            self.add_neighbor_previous_value(sender, data['previous_value'])

            if shared_exp:
                self.experience_buffer.add(
                    exp=[dict_to_state(shared_exp[0]), dict_to_state(shared_exp[1])],
                )
        elif message_type == InfoSharingType.BURIED_HUMAN_SHARING:
            self.log.info(f'Shared buried data: {data}')

    def __call__(self, *args, **kwargs):
        self.log.info(f'Initializing agent {self.agent_id}')
        while not self._terminate:
            self.comm.listen_to_network()
            self.graph.connect()
            self.dcop.resolve_value()
        self.log.info(f'Agent {self.agent_id} is shutting down. Adios!')

    def record_deliberation_time(self, t, val):
        self.dcop.record_agent_metric('deliberation time', t, val)

    def handle_busy_agent(self, message):
        data = message['payload']
        sender = data['agent_id']
        self.log.info(f'Received Busy message from {sender}')

        if sender in self.agents_in_comm_range:
            self.agents_in_comm_range.remove(sender)

        if len(self.agents_in_comm_range) == 0 and not self.value:
            self.comm.threadsafe_execution(self.dcop.select_random_value)

    def send_busy_to_neighbors(self):
        if self.agents_in_comm_range:
            self.log.info(f'Sending Busy message to {self.agents_in_comm_range}')
            self.comm.threadsafe_execution(self._send_busy_msgs)

    def _send_busy_msgs(self):
        for agt in self.agents_in_comm_range:
            self.comm.send_busy_message(agt, {'agent_id': self.agent_id})

