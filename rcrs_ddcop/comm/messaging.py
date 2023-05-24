import datetime
from enum import auto
import json

from strenum import StrEnum


class AgentMsgTypes(StrEnum):
    TEST = auto()
    SHARED_INFO = auto()


class DIGCAMsgTypes(StrEnum):
    ANNOUNCE = auto()
    ADD_ME = auto()
    ANNOUNCE_RESPONSE_IGNORED = auto()
    ANNOUNCE_RESPONSE = auto()
    CHILD_ADDED = auto()
    ALREADY_ACTIVE = auto()
    PARENT_ASSIGNED = auto()
    PARENT_ALREADY_ASSIGNED = auto()


class DPOPMsgTypes(StrEnum):
    REQUEST_UTIL_MESSAGE = auto()
    VALUE_MESSAGE = auto()
    UTIL_MESSAGE = auto()


class CoCoAMsgTypes(StrEnum):
    UPDATE_STATE_MESSAGE = auto()
    INQUIRY_MESSAGE = auto()
    COST_MESSAGE = auto()
    EXECUTION_REQUEST = auto()


def _create_msg(msg_type: str, data: dict):
    return json.dumps({
        'type': msg_type,
        'payload': data,
        'timestamp': datetime.datetime.now().timestamp()
    })


def create_test_message(data: dict):
    return _create_msg(AgentMsgTypes.TEST, data)


# print(create_test_message({'p': 1, 'a': 2}))


def create_announce_message(data):
    return _create_msg(DIGCAMsgTypes.ANNOUNCE, data)


def create_add_me_message(data):
    return _create_msg(DIGCAMsgTypes.ADD_ME, data)


def create_announce_response_ignored_message(data):
    return _create_msg(DIGCAMsgTypes.ANNOUNCE_RESPONSE_IGNORED, data)


def create_announce_response_message(data):
    return _create_msg(DIGCAMsgTypes.ANNOUNCE_RESPONSE, data)


def create_child_added_message(data):
    return _create_msg(DIGCAMsgTypes.CHILD_ADDED, data)


def create_already_active_message(data):
    return _create_msg(DIGCAMsgTypes.ALREADY_ACTIVE, data)


def create_parent_assigned_message(data):
    return _create_msg(DIGCAMsgTypes.PARENT_ASSIGNED, data)


def create_parent_already_assigned_message(data):
    return _create_msg(DIGCAMsgTypes.PARENT_ALREADY_ASSIGNED, data)


def create_request_util_message(data):
    return _create_msg(DPOPMsgTypes.REQUEST_UTIL_MESSAGE, data)


def create_value_message(data):
    return _create_msg(DPOPMsgTypes.VALUE_MESSAGE, data)


def create_util_message(data):
    return _create_msg(DPOPMsgTypes.UTIL_MESSAGE, data)


def create_update_state_message(data):
    return _create_msg(CoCoAMsgTypes.UPDATE_STATE_MESSAGE, data)


def create_inquiry_message(data):
    return _create_msg(CoCoAMsgTypes.INQUIRY_MESSAGE, data)


def create_cost_message(data):
    return _create_msg(CoCoAMsgTypes.COST_MESSAGE, data)


def create_execution_request_message(data):
    return _create_msg(CoCoAMsgTypes.EXECUTION_REQUEST, data)


def create_shared_info_message(data):
    return _create_msg(AgentMsgTypes.SHARED_INFO, data)
