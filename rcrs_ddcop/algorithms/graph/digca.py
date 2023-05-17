import enum
import random
import time

from rcrs_ddcop.algorithms.graph import DynaGraph

MAX_OUT_DEGREE = 3


class State(enum.Enum):
    ACTIVE = enum.auto()
    INACTIVE = enum.auto()


class DIGCA(DynaGraph):
    """
    Implementation of the Dynamic Interaction Graph Construction algorithm
    """

    def __init__(self,  agent):
        super(DIGCA, self).__init__(agent)
        self._has_sent_parent_available = False
        self.pinged_list_dict = {}
        self.state = State.INACTIVE
        self.announceResponseList = []
        self._ignored_ann_msgs = {}
        self._parent_already_assigned_msgs = {}
        self._timeout_delay_in_seconds = .5
        self._timeout_delay_start = None

    def on_time_step_changed(self):
        self._ignored_ann_msgs.clear()
        self._parent_already_assigned_msgs.clear()
        self._has_sent_parent_available = False
        self._timeout_delay_start = time.time()
        self.exec_started = False

    def connect(self):
        if not self.parent and self.has_potential_parent() and self.state == State.INACTIVE:
            self.log.debug(f'Publishing Announce message...')

            # publish Announce message
            self.comm.broadcast_announce_message(self.agent.agents_in_range)

            # wait to receive responses
            self.comm.listen_to_network()

            self.log.debug(f'AnnounceResponse list in connect: {self.announceResponseList}')

            # select agent to connect to
            selected_agent = None
            if self.announceResponseList:
                selected_agent = random.choice(self.announceResponseList)

            if selected_agent is not None:
                self.log.debug(f'Selected agent for AddMe: {selected_agent}')
                self.comm.send_add_me_message(selected_agent)
                self.state = State.ACTIVE

            # send announce response ignored messages
            for a in set(self.announceResponseList):
                if a != selected_agent:
                    self.comm.send_announce_response_ignored_message(a)

            self.announceResponseList.clear()

        elif not self.exec_started \
                and self._timeout_delay_start \
                and time.time() - self._timeout_delay_start > self._timeout_delay_in_seconds:
            self.start_dcop()
            self._timeout_delay_start = None

    def receive_announce(self, message):
        self.log.debug(f'Received announce: {message}')
        sender = message['payload']['agent_id']

        if self.state == State.INACTIVE and self.agent.agent_id < sender:
            self.comm.send_announce_response(sender)

    def receive_announce_response(self, message):
        self.log.debug(f'Received announce response: {message}')
        sender = message['payload']['agent_id']

        if self.state == State.INACTIVE:
            self.announceResponseList.append(sender)
            self.log.debug(f'AnnounceResponse list: {self.announceResponseList}')

    def receive_add_me(self, message):
        self.log.debug(f'Received AddMe: {message}')
        sender = message['payload']['agent_id']

        if self.state == State.INACTIVE and len(self.children) < MAX_OUT_DEGREE:
            self.children.append(sender)
            self.comm.send_child_added_message(sender)
            self.log.info(f'Added agent {sender} to children: {self.children}')

            # update current graph
            # self.channel.basic_publish(
            #     exchange=messaging.COMM_EXCHANGE,
            #     routing_key=f'{messaging.SIM_ENV_CHANNEL}',
            #     body=messaging.create_add_graph_edge_message({
            #         'agent_id': self.agent.agent_id,
            #         'from': self.agent.agent_id,
            #         'to': sender,
            #     })
            # )

            # inform dashboard about the connection
            # self.report_connection(parent=self.agent.agent_id, child=sender, constraint=constraint)
        else:
            self.log.debug(f'Rejected AddMe from agent: {sender}, sending AlreadyActive message')
            self.comm.send_already_active_message(self.agent.agent_id)

    def receive_child_added(self, message):
        self.log.debug(f'Received ChildAdded: {message}')
        sender = message['payload']['agent_id']

        if self.state == State.ACTIVE and not self.parent:
            self.state = State.INACTIVE
            self.parent = sender
            self.comm.send_parent_assigned_message(sender)
            self.log.info(f'Set parent node to agent {sender}')

            # update current graph
            # self.channel.basic_publish(
            #     exchange=messaging.COMM_EXCHANGE,
            #     routing_key=f'{messaging.SIM_ENV_CHANNEL}',
            #     body=messaging.create_add_graph_edge_message({
            #         'agent_id': self.agent.agent_id,
            #         'from': sender,
            #         'to': self.agent.agent_id,
            #     })
            # )

            if self.agent.graph_traversing_order == 'bottom-up':
                self.start_dcop()

    def receive_parent_assigned(self, message):
        self.log.debug(f'Received ParentAssigned: {message}')

        if self.agent.graph_traversing_order == 'top-down':
            self.start_dcop()

    def receive_already_active(self, message):
        self.log.debug(f'Received AlreadyActive: {message}')
        self.state = State.INACTIVE

    def receive_announce_response_ignored(self, message):
        sender = message['payload']['agent_id']
        self._ignored_ann_msgs[sender] = message
        self.log.info(f'Received announce ignored message from {sender}')

        if len(set(self._ignored_ann_msgs)) == len(self._get_potential_children()) and not self.agent.value:
            self.start_dcop()

    def receive_parent_available_message(self, message):
        # self.log.info(f'Received parent available message: {message}')
        if self.parent:
            sender = message['payload']['agent_id']
            self.comm.send_parent_already_assigned_message(sender)

    def receive_parent_already_assigned(self, message):
        sender = message['payload']['agent_id']
        self._parent_already_assigned_msgs[sender] = message
        self.log.debug(f'Received parent already assigned message from {sender}')

        if len(self._parent_already_assigned_msgs.keys()) == len(self._get_potential_children()) \
                and not self.agent.value:
            self._has_sent_parent_available = True
            self.start_dcop()

    def _get_potential_children(self):
        agents = []
        for _agt in set(self.agent.new_agents) - set(self.neighbors):
            if int(_agt.replace('a', '')) > int(self.agent.agent_id.replace('a', '')):
                agents.append(_agt)

        return agents

    def has_potential_parent(self):
        for _agt in set(self.agent.new_agents) - set(self.neighbors):
            if int(_agt.replace('a', '')) < int(self.agent.agent_id.replace('a', '')):
                return True

        return False

    def has_potential_child(self):
        return bool(self._get_potential_children())

    def has_potential_neighbor(self):
        return self.has_potential_child() or (not self.parent and self.has_potential_parent())
