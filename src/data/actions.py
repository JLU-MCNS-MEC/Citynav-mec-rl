"""Official CityNav six-action interface.

Action IDs and execution parameters follow
``water-cookie/citynav/vlnce/actions.py``:

======  =================  =========================
ID      Action             Execution
======  =================  =========================
0       STOP               no movement
1       MOVE_FORWARD       5 m forward
2       TURN_RIGHT         30 degrees right
3       TURN_LEFT          30 degrees left
4       GO_UP              2 m up
5       GO_DOWN            2 m down
======  =================  =========================

The MEC movement head predicts the five non-stop actions in the order of
``MOVE_ACTION_IDS`` (Forward, Right, Left, Up, Down).  The binary arrival head
predicts STOP.  This module is the single source of truth for that ordering.
"""

from __future__ import annotations

from enum import IntEnum


class DiscreteAction(IntEnum):
    STOP = 0
    MOVE_FORWARD = 1
    TURN_RIGHT = 2
    TURN_LEFT = 3
    GO_UP = 4
    GO_DOWN = 5


STOP = DiscreteAction.STOP
MOVE_FORWARD = DiscreteAction.MOVE_FORWARD
TURN_RIGHT = DiscreteAction.TURN_RIGHT
TURN_LEFT = DiscreteAction.TURN_LEFT
GO_UP = DiscreteAction.GO_UP
GO_DOWN = DiscreteAction.GO_DOWN

ACTION_ORDER = (
    STOP,
    MOVE_FORWARD,
    TURN_RIGHT,
    TURN_LEFT,
    GO_UP,
    GO_DOWN,
)
MOVE_ACTION_IDS = (
    MOVE_FORWARD,
    TURN_RIGHT,
    TURN_LEFT,
    GO_UP,
    GO_DOWN,
)
MOVE_HEAD_INDEX = {
    action: index for index, action in enumerate(MOVE_ACTION_IDS)
}
ACTION_NAMES = {
    STOP: "Stop",
    MOVE_FORWARD: "MoveForward",
    TURN_RIGHT: "TurnRight",
    TURN_LEFT: "TurnLeft",
    GO_UP: "GoUp",
    GO_DOWN: "GoDown",
}

NUM_ACTIONS = len(ACTION_ORDER)
NUM_MOVE_ACTIONS = len(MOVE_ACTION_IDS)

# Fixed execution parameters from the official implementation.
FORWARD_STEP_M = 5.0
TURN_STEP_DEG = 30.0
VERTICAL_STEP_M = 2.0


def action_name(action: int) -> str:
    return ACTION_NAMES[DiscreteAction(int(action))]


def move_head_index(action: int) -> int:
    """Map a non-stop action ID to its index in the five-way movement head."""
    return MOVE_HEAD_INDEX[DiscreteAction(int(action))]


def move_action_from_index(index: int) -> int:
    """Map a five-way movement-head index to an official action ID."""
    return int(MOVE_ACTION_IDS[int(index)])
