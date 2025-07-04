import datetime
from enum import auto
import json

from strenum import StrEnum


class AgentMsgTypes(StrEnum):
    TEST = auto()
    BUSY = auto()
    BUILDING_METRICS = auto()
    TRAINING_METRICS = auto()


class DIGCAMsgTypes(StrEnum):
    ANNOUNCE = auto()
    ADD_ME = auto()
    PSEUDO_PARENT_REQEUST = auto()
    ANNOUNCE_RESPONSE = auto()
    CHILD_ADDED = auto()
    ALREADY_ACTIVE = auto()
    PARENT_ASSIGNED = auto()
    PARENT_ALREADY_ASSIGNED = auto()
    PSEUDO_CHILD_ADDED = auto()
    SEPARATOR = auto()


class InfoSharing(StrEnum):
    EXP_HISTORY_DISCLOSURE = auto()
    EXP_SHARING_WITH_REQUEST = auto()
    EXP_SHARING = auto()
    NEIGHBOR_UPDATE = auto()


class DPOPMsgTypes(StrEnum):
    REQUEST_UTIL_MESSAGE = auto()
    DPOP_VALUE_MESSAGE = auto()
    UTIL_MESSAGE = auto()


class CoCoAMsgTypes(StrEnum):
    UPDATE_STATE_MESSAGE = auto()
    INQUIRY_MESSAGE = auto()
    COST_MESSAGE = auto()
    EXECUTION_REQUEST = auto()
    CoCoA_VALUE_MESSAGE = auto()


class LSLAMsgTypes(StrEnum):
    LSLA_INQUIRY_MESSAGE = auto()
    LSLA_UTIL_MESSAGE = auto()


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


def create_pseudo_parent_reqeust_message(data):
    return _create_msg(DIGCAMsgTypes.PSEUDO_PARENT_REQEUST, data)


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


def create_separator_message(data):
    return _create_msg(DIGCAMsgTypes.SEPARATOR, data)


def create_request_util_message(data):
    return _create_msg(DPOPMsgTypes.REQUEST_UTIL_MESSAGE, data)


def create_dpop_value_message(data):
    return _create_msg(DPOPMsgTypes.DPOP_VALUE_MESSAGE, data)


def create_util_message(data):
    return _create_msg(DPOPMsgTypes.UTIL_MESSAGE, data)


def create_update_state_message(data):
    return _create_msg(CoCoAMsgTypes.UPDATE_STATE_MESSAGE, data)


def create_cocoa_value_message(data):
    return _create_msg(CoCoAMsgTypes.CoCoA_VALUE_MESSAGE, data)


def create_inquiry_message(data):
    return _create_msg(CoCoAMsgTypes.INQUIRY_MESSAGE, data)


def create_cost_message(data):
    return _create_msg(CoCoAMsgTypes.COST_MESSAGE, data)


def create_execution_request_message(data):
    return _create_msg(CoCoAMsgTypes.EXECUTION_REQUEST, data)


def create_lsla_inquiry_message(data):
    return _create_msg(LSLAMsgTypes.LSLA_INQUIRY_MESSAGE, data)


def create_lsla_util_message(data):
    return _create_msg(LSLAMsgTypes.LSLA_UTIL_MESSAGE, data)


def create_busy_message(data):
    return _create_msg(AgentMsgTypes.BUSY, data)


def create_pseudo_child_added_message(data):
    return _create_msg(DIGCAMsgTypes.PSEUDO_CHILD_ADDED, data)


def create_exp_history_disclosure_message(data):
    return _create_msg(InfoSharing.EXP_HISTORY_DISCLOSURE, data)


def create_exp_sharing_with_request_message(data):
    return _create_msg(InfoSharing.EXP_SHARING_WITH_REQUEST, data)


def create_exp_sharing_message(data):
    return _create_msg(InfoSharing.EXP_SHARING, data)


def create_neighbor_update_message(data):
    return _create_msg(InfoSharing.NEIGHBOR_UPDATE, data)


def create_building_metrics_message(data):
    return _create_msg(AgentMsgTypes.BUILDING_METRICS, data)

def create_training_metrics_message(data):
    return _create_msg(AgentMsgTypes.TRAINING_METRICS, data)
