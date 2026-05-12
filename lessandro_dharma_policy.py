"""Among Them policy: lessandro-dharma.

Buddhist theme: dharma = righteous path, duty, cosmic order.
Strategy: navigate purposefully toward task indicators, complete tasks as
crewmate; as imposter exploit positioning intelligently.

Observation format: (batch, 4, 128, 128) uint8, PICO-8 color indices 0-15.
Actions: indices into BITWORLD_ACTION_NAMES (27 total).
"""

from __future__ import annotations

import numpy as np

from mettagrid.bitworld import (
    BITWORLD_ACTION_NAMES,
    SCREEN_HEIGHT,
    SCREEN_WIDTH,
    bitworld_action_index,
    bitworld_action_name,
    encode_buttons,
)
from mettagrid.policy.policy import AgentPolicy, MultiAgentPolicy
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.simulator import Action, AgentObservation

# PICO-8 palette indices
# 0=black, 1=dark-blue, 2=dark-purple, 3=dark-green, 4=brown, 5=dark-gray
# 6=light-gray, 7=white, 8=red, 9=orange, 10=yellow, 11=bright-green
# 12=blue, 13=indigo, 14=pink, 15=peach

# Bright task-indicator colors — tasks in Among Them are highlighted with
# vivid hues against the dark map background.
TASK_COLORS = frozenset([9, 10, 11, 14, 15])  # orange, yellow, green, pink, peach
# Floor/navigable colors (not walls)
FLOOR_COLORS = frozenset([5, 6, 7])  # dark-gray, light-gray, white
# Wall / void colors (avoid)
WALL_COLORS = frozenset([0, 1, 2])  # black, dark-blue, dark-purple

# Quadrant slices for quick directional scoring
CENTER_R = SCREEN_HEIGHT // 2
CENTER_C = SCREEN_WIDTH // 2

# Interaction window around screen center
INTERACT_HALF = 12

# Action index cache
_NOOP = bitworld_action_index(encode_buttons(()))
_UP = bitworld_action_index(encode_buttons(("up",)))
_DOWN = bitworld_action_index(encode_buttons(("down",)))
_LEFT = bitworld_action_index(encode_buttons(("left",)))
_RIGHT = bitworld_action_index(encode_buttons(("right",)))
_A = bitworld_action_index(encode_buttons(("a",)))
_SELECT = bitworld_action_index(encode_buttons(("select",)))

_DPAD = [_UP, _DOWN, _LEFT, _RIGHT]
_SPIRAL = [_RIGHT, _RIGHT, _DOWN, _DOWN, _LEFT, _LEFT, _UP, _UP]

# Exploration phases: how many ticks to hold each direction
EXPLORE_HOLD = 10
# How often to attempt an A-press even during exploration
A_PRESS_INTERVAL = 30
# How often SELECT is pressed to report / vote
SELECT_INTERVAL = 500


class AmongThemAgentPolicy(AgentPolicy):
    def __init__(self, policy_env_info: PolicyEnvInterface, parent: "AmongThemPolicy", agent_id: int):
        super().__init__(policy_env_info)
        self._parent = parent
        self._agent_id = agent_id

    def step(self, obs: AgentObservation) -> Action:
        del obs
        return Action(name=bitworld_action_name(self._parent.next_action(self._agent_id)))


class AmongThemPolicy(MultiAgentPolicy):
    """Dharma policy — task-seeking with pixel-based heuristics.

    Per-step logic:
    1. Look at the most recent frame in the stack.
    2. If task-colored pixels are near screen center → press A.
    3. Else move toward the quadrant with the most task-colored pixels.
    4. Fall back to a systematic exploration spiral.
    5. Occasionally press SELECT to trigger reports / voting.
    """

    def __init__(self, policy_env_info: PolicyEnvInterface, device: str = "cpu"):
        super().__init__(policy_env_info, device=device)
        if tuple(policy_env_info.action_names) != BITWORLD_ACTION_NAMES:
            raise ValueError("AmongThemPolicy requires the BitWorld AmongThem action space")
        self._tick = 0
        self._agent_steps_this_tick = 0

    # ------------------------------------------------------------------
    # Batch path (tournament / evaluation)
    # ------------------------------------------------------------------

    def step_batch(self, raw_observations: np.ndarray, raw_actions: np.ndarray) -> None:
        raw_actions[...] = self._choose_actions(raw_observations)
        self._tick += 1
        self._agent_steps_this_tick = 0

    def _choose_actions(self, raw_observations: np.ndarray) -> np.ndarray:
        """Return action indices for every agent in the batch.

        raw_observations: (batch, frame_stack, H, W) uint8 with PICO-8 indices.
        """
        batch_size = raw_observations.shape[0]
        actions = np.empty(batch_size, dtype=np.int32)

        # Use the most recent frame in the stack
        if raw_observations.ndim == 4:
            frames = raw_observations[:, -1]  # (batch, H, W)
        else:
            # Fallback: flat or 2-D array — reshape to (batch, H, W)
            frames = raw_observations.reshape(batch_size, SCREEN_HEIGHT, SCREEN_WIDTH)

        for i in range(batch_size):
            actions[i] = self._pick_action(frames[i], agent_id=i)

        return actions

    def _pick_action(self, frame: np.ndarray, agent_id: int) -> int:
        """Pick an action for a single agent given its current frame."""

        # --- Task color mask ---
        task_mask = np.isin(frame, list(TASK_COLORS))

        # 1. Interact if task pixels are near screen center
        center_task_count = task_mask[
            CENTER_R - INTERACT_HALF : CENTER_R + INTERACT_HALF,
            CENTER_C - INTERACT_HALF : CENTER_C + INTERACT_HALF,
        ].sum()
        if center_task_count > 15:
            return _A

        # 2. Periodic SELECT to report bodies / open voting
        if (self._tick + agent_id * 37) % SELECT_INTERVAL < 3:
            return _SELECT

        # 3. Move toward the quadrant richest in task colors
        quadrant_scores = {
            _UP: int(task_mask[: CENTER_R - 4, :].sum()),
            _DOWN: int(task_mask[CENTER_R + 4 :, :].sum()),
            _LEFT: int(task_mask[:, : CENTER_C - 4].sum()),
            _RIGHT: int(task_mask[:, CENTER_C + 4 :].sum()),
        }
        best_action, best_score = max(quadrant_scores.items(), key=lambda kv: kv[1])
        if best_score > 8:
            return best_action

        # 4. Periodic A-press to catch tasks missed by vision
        if self._tick % A_PRESS_INTERVAL == 0:
            return _A

        # 5. Exploration spiral (deterministic per agent to spread out)
        spiral_index = (self._tick // EXPLORE_HOLD + agent_id * 3) % len(_SPIRAL)
        return _SPIRAL[spiral_index]

    # ------------------------------------------------------------------
    # Per-agent path (interactive / per-agent runner)
    # ------------------------------------------------------------------

    def agent_policy(self, agent_id: int) -> AgentPolicy:
        return AmongThemAgentPolicy(self._policy_env_info, self, agent_id)

    def next_action(self, agent_id: int) -> int:
        """Called by AmongThemAgentPolicy.step() — tick managed externally."""
        self._agent_steps_this_tick += 1
        if self._agent_steps_this_tick == self._policy_env_info.num_agents:
            self._agent_steps_this_tick = 0
            self._tick += 1
        # Use zero-observation action selection with spiral fallback
        spiral_index = (self._tick // EXPLORE_HOLD + agent_id * 3) % len(_SPIRAL)
        return _SPIRAL[spiral_index]
