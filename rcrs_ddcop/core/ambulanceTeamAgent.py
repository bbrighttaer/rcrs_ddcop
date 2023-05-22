import threading
from typing import List

from rcrs_core.agents.agent import Agent
from rcrs_core.connection import URN
from rcrs_core.constants import kernel_constants
from rcrs_core.entities.ambulanceTeam import AmbulanceTeamEntity
from rcrs_core.entities.civilian import Civilian
from rcrs_core.entities.entity import Entity
from rcrs_core.worldmodel.entityID import EntityID

from rcrs_ddcop.core.bdi_agent import BDIAgent

SEARCH_ACTION = -1
UNLOAD_CIVILIAN = -2
NO_OP = -3


class AmbulanceTeamAgent(Agent):
    def __init__(self, pre):
        Agent.__init__(self, pre)
        self.name = 'AmbulanceTeamAgent'
        self.bdi_agent = None

    def precompute(self):
        self.Log.info('precompute finished')

    def post_connect(self):
        super(AmbulanceTeamAgent, self).post_connect()
        threading.Thread(target=self._start_bdi_agent, daemon=True).start()

    def _start_bdi_agent(self):
        # create BDI agent after RCRS agent is setup
        self.bdi_agent = BDIAgent(self)
        self.bdi_agent()

    def get_requested_entities(self):
        return [URN.Entity.AMBULANCE_TEAM]

    def get_change_set_entities(self, entity_ids: List[EntityID]):
        entities = []
        for e_id in entity_ids:
            entity = self.world_model.get_entity(e_id)
            if entity:
                entities.append(entity)
        return entities

    def get_agents_in_comm_range(self, entities: List[Entity]):
        neighbors = []
        for entity in entities:
            if isinstance(entity, AmbulanceTeamEntity) and entity.entity_id != self.agent_id:
                neighbors.append(entity.entity_id.get_value())
        return neighbors

    def get_civilians(self, entities: List[Entity]):
        civilians = []
        for entity in entities:
            if isinstance(entity, Civilian):
                civilians.append(entity.entity_id.get_value())
        return civilians

    def think(self, time_step, change_set, heard):
        self.Log.info(f'Time step {time_step}')
        if time_step == self.config.get_value(kernel_constants.IGNORE_AGENT_COMMANDS_KEY):
            self.send_subscribe(time_step, [1, 2])

        # get visible entities
        change_set_entities = self.get_change_set_entities(list(change_set.changed.keys()))

        # get agents in communication range
        neighbors = self.get_agents_in_comm_range(change_set_entities)
        self.bdi_agent.agents_in_comm_range = neighbors

        # get civilians to construct domain
        civilians = self.get_civilians(change_set_entities)
        self.bdi_agent.domain = [SEARCH_ACTION, UNLOAD_CIVILIAN] + civilians

        # information sharing
        self.bdi_agent.share_information()

        # execute thinking process
        action = self.bdi_agent.deliberate()
        action_lbl = {
            SEARCH_ACTION: 'search',
            UNLOAD_CIVILIAN: 'unload',
        }.get(action, f'rescue civilian {action}')
        self.Log.debug(f'Selected action: {action_lbl}')

        # send action to environment
        if action is None or action == SEARCH_ACTION:
            if action is None:
                self.Log.warning(f'No action selected, selecting search action instead')
            my_path = self.random_walk()
            self.send_move(time_step, my_path)
        elif action == UNLOAD_CIVILIAN:
            self.send_unload(time_step)
        else:
            self.send_load(time_step, EntityID(action))

        # self.send_load(time_step, target)
        # self.send_unload(time_step)
        # self.send_say(time_step, 'HELP')
        # self.send_speak(time_step, 'HELP meeeee police', 1)
        # self.send_move(time_step, my_path)
        # self.send_rest(time_step)
