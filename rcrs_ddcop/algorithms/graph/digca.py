import enum
import time
from collections import deque
from functools import partial

import numpy as np

from rcrs_ddcop.algorithms.graph import DynaGraph
from rcrs_ddcop.core.enums import DynamicGraphCallback

MAX_OUT_DEGREE = 3


class State(enum.Enum):
    ACTIVE = enum.auto()
    INACTIVE = enum.auto()


class DIGCA(DynaGraph):
    """
    Implementation of the Dynamic Interaction Graph Construction algorithm
    """

    def __init__(self,  agent, timeout: float, max_num_of_neighbors: int = -1):
        super(DIGCA, self).__init__(agent)
        self.pinged_list_dict = {}
        self.state = State.INACTIVE
        self.separator = deque()
        self.announceResponseList = deque()
        self._pseudo_parent_request_mgs = {}
        self._parent_already_assigned_msgs = {}
        self._timeout_delay_in_seconds = timeout
        self._timeout_delay_start = None
        self._sent_announce_msg_list = deque()
        self._pseudo_parent_req_msg_list = deque()
        self._max_num_neighbors = max_num_of_neighbors

    def on_time_step_changed(self):
        self._pseudo_parent_request_mgs.clear()
        self._parent_already_assigned_msgs.clear()
        self._timeout_delay_start = time.time()
        self._pseudo_parent_req_msg_list.clear()
        self._sent_announce_msg_list.clear()
        self.announceResponseList.clear()
        self.exec_started = False

    def connect(self):
        parents = self.get_potential_parents()
        targets = set(parents) - set(self._sent_announce_msg_list)
        if targets and not self.parent and parents:
            self.log.debug(f'Publishing Announce message...')

            # publish Announce message
            if targets:
                self._sent_announce_msg_list.extend(targets)
                self.log.debug(f'Sending Announce messages to {targets}')
                self.comm.broadcast_announce_message(targets)

        # period timeout check
        # elif self.is_connection_timeout():
        #     self.log.warning(f'DIGCA timeout, starting D-DCOP.')
        #     self.start_dcop()
        #     self._timeout_delay_start = None

        else:
            for a in parents:
                invalid_pp = self._pseudo_parent_req_msg_list + self._sent_announce_msg_list + self.all_neighbors
                if a not in invalid_pp:
                    self.send_pseudo_parent_request_message(a)

    def send_pseudo_parent_request_message(self, a):
        self.log.debug(f'Sending pseudo-parent request to {a}')
        self.comm.send_pseudo_parent_request_message(a, domain=self.agent.domain)
        self._pseudo_parent_req_msg_list.append(a)

    def is_connection_timeout(self):
        return not self.agent.value \
            and not self.exec_started \
            and self._timeout_delay_start \
            and time.time() - self._timeout_delay_start > self._timeout_delay_in_seconds

    def send_connection_requests(self):
        self.log.debug(f'AnnounceResponse list in connect: {self.announceResponseList}')
        # select agent to connect to
        selected_agent = None
        agts = []
        p_list = []

        if self.announceResponseList:
            factors = []
            for a in self.announceResponseList:
                num_neighbors = a[1]
                if num_neighbors < self._max_num_neighbors:
                    agts.append(a[0])
                    factors.append(num_neighbors)
                else:
                    p_list.append(a[0])

            if agts:
                factors = np.array(factors) + 1e-10  # (np.max(factors) - np.array(factors)) + 1e-10
                probs = factors / np.sum(factors)
                selected_agent = np.random.choice(agts, p=probs)

        if selected_agent is not None:
            self.log.debug(f'Selected agent for AddMe: {selected_agent}')
            self.comm.send_add_me_message(selected_agent, domain=self.agent.domain)
            self.state = State.ACTIVE

        # send pseudo-parent request message
        for a in set(agts + p_list):
            if a != selected_agent:
                self.send_pseudo_parent_request_message(a)
        self.announceResponseList.clear()

    def receive_announce(self, message):
        self.log.debug(f'Received announce: {message}')
        sender = message['payload']['agent_id']

        if self.agent.agent_id < sender:
            self.comm.send_announce_response(sender, len(self.children))

    def receive_announce_response(self, message):
        self.log.debug(f'Received announce response: {message}')
        sender = message['payload']['agent_id']
        num_of_children = message['payload']['num_of_children']

        if sender not in self.announceResponseList:
            self.announceResponseList.append([sender, num_of_children])
            if len(set(self._sent_announce_msg_list) - set([a[0] for a in self.announceResponseList])) == 0:
                self.send_connection_requests()

    def receive_add_me(self, message):
        self.log.debug(f'Received AddMe: {message}')
        sender = message['payload']['agent_id']

        if sender not in self.children:
            self.children.append(sender)
            if sender in self.pseudo_children:
                self.pseudo_children.remove(sender)
            self.agent.add_neighbor_domain(sender, message['payload']['domain'])
            self.comm.send_child_added_message(sender, domain=self.agent.domain)
            self.log.debug(f'Added agent {sender} to children: {self.children}')

            # callbacks
            self.fire_callbacks(
                cb_types=[DynamicGraphCallback.CHILD_ADDED, DynamicGraphCallback.AGENT_CONNECTED],
                agent=sender,
            )
        # else:
        #     self.log.debug(f'Rejected AddMe from agent: {sender}, sending AlreadyActive message')
        #     self.comm.send_already_active_message(sender)

    def receive_child_added(self, message):
        self.log.debug(f'Received ChildAdded: {message}')
        sender = message['payload']['agent_id']

        if not self.parent:
            self.state = State.INACTIVE
            self.parent = sender
            if sender in self.pseudo_parents:
                self.pseudo_parents.remove(sender)
            self.agent.add_neighbor_domain(sender, message['payload']['domain'])
            self.comm.send_parent_assigned_message(sender)
            self._sent_announce_msg_list.remove(sender)
            self.log.debug(f'Set parent node to agent {sender}')

            # callbacks
            self.fire_callbacks(
                cb_types=[DynamicGraphCallback.PARENT_ASSIGNED, DynamicGraphCallback.AGENT_CONNECTED],
                agent=sender,
            )

            if self.agent.graph_traversing_order == 'bottom-up':
                self.start_dcop()

    def receive_parent_assigned(self, message):
        self.log.debug(f'Received ParentAssigned: {message}')

        if self.agent.graph_traversing_order == 'top-down':
            self.start_dcop()

    def receive_already_active(self, message):
        sender = message['payload']['agent_id']
        self.log.debug(f'Received AlreadyActive: {message}')
        self.state = State.INACTIVE

        # remove the sender from the Announce msg received list so that the sender can receive Announce msg again
        self._sent_announce_msg_list.remove(sender)

    def receive_pseudo_parent_request(self, message):
        sender = message['payload']['agent_id']
        self._pseudo_parent_request_mgs[sender] = message
        self.log.debug(f'Received pseudo parent request message from {sender}')

        # add sender as pseudo-child
        if sender not in self.all_children:
            self.pseudo_children.append(sender)
            self.agent.add_neighbor_domain(sender, message['payload']['domain'])
            self.comm.send_pseudo_child_added_message(sender, domain=self.agent.domain)
            self.log.debug(f'Agent {sender} added as pseudo child successfully')

            # callbacks
            self.fire_callbacks(
                cb_types=[DynamicGraphCallback.PSEUDO_CHILD_ADDED, DynamicGraphCallback.AGENT_CONNECTED],
                agent=sender,
            )

            self.start_dcop()

    def receive_pseudo_child_added_message(self, message):
        sender = message['payload']['agent_id']
        self._parent_already_assigned_msgs[sender] = message
        self.log.debug(f'Received pseudo child added message from {sender}')

        # add sender as pseudo-parent
        if sender not in self.pseudo_parents:
            self.pseudo_parents.append(sender)
            self.agent.add_neighbor_domain(sender, message['payload']['domain'])
            self.log.debug(f'Agent {sender} added to pseudo parents successfully')

            if sender in self._sent_announce_msg_list:
                self._sent_announce_msg_list.remove(sender)

            # callbacks
            self.fire_callbacks(
                cb_types=[DynamicGraphCallback.PSEUDO_PARENT_ADDED, DynamicGraphCallback.AGENT_CONNECTED],
                agent=sender,
            )

            self.start_dcop()

    def receive_parent_available_message(self, message):
        # self.log.debug(f'Received parent available message: {message}')
        if self.parent:
            sender = message['payload']['agent_id']
            self.comm.send_parent_already_assigned_message(sender)

    def receive_parent_already_assigned(self, message):
        sender = message['payload']['agent_id']
        self._parent_already_assigned_msgs[sender] = message
        self.log.debug(f'Received parent already assigned message from {sender}')

        self.start_dcop()

    def receive_separator_message(self, message):
        sender = message['payload']['agent_id']
        separator = message['payload']['separator']
        self.log.debug(f'Received separator message from {sender}: {separator}')
        self.update_separator(separator + [sender])

    def update_separator(self, sep: list):
        self.separator = deque(sep)
        for child in self.children:
            self.comm.threadsafe_execution(
                partial(self.comm.send_separator_message, child, separator=list(self.separator))
            )

    def get_potential_children(self):
        agents = []
        for _agt in self.agent.new_agents:
            if _agt > self.agent.agent_id:
                agents.append(_agt)

        return agents

    def get_potential_parents(self):
        agents = []
        for _agt in self.agent.new_agents:
            if _agt < self.agent.agent_id:
                agents.append(_agt)

        return agents

    def has_potential_parent(self):
        for _agt in self.agent.new_agents:
            if _agt < self.agent.agent_id:
                return True

        return False

    def has_potential_child(self):
        return bool(self.get_potential_children())

    def has_potential_neighbor(self):
        return self.has_potential_child() or (not self.parent and self.has_potential_parent())

    def remove_agent(self, agent):
        if self.parent == agent:
            self.state = State.INACTIVE
            self.parent = None
        elif agent in self.children:
            self.children.remove(agent)
        elif agent in self.pseudo_parents:
            self.pseudo_parents.remove(agent)
        elif agent in self.pseudo_children:
            self.pseudo_children.remove(agent)

        self.agent.remove_neighbor_domain(agent)

        self.report_agent_disconnection(agent)
