"""Among Them policy: lessandro-anatta.

Buddhist theme: anatta = non-self.  We act for the collective, not the ego.
Strategy: use sprite_player structured observations for optimal play.

Observation layout (sprite_player mode, 1231 features):
  [0]        phase         (LOBBY=0 PLAYING=1 VOTING=2 VOTE_RESULT=3 GAME_OVER=4 ROLE_REVEAL=5)
  [1]        kill_icon     (0=crewmate, 1=imposter-cooldown, 255=imposter-ready)
  [2]        task_progress (0-255 scaled % of all tasks done)
  [3]        tasks_remaining (count)
  [4:1028]   map grid 32×32 PICO-8 colour at each cell
  [1028:1092] player slots 0-15, 4 bytes each: x, y, colour, flags
              flags: PRESENT=1, ALIVE=4, FLIP_H=16, GHOST=32
  [1092:1156] body slots 0-15, 4 bytes each: x, y, colour, 1
  [1156:1231] task slots 0-14, 5 bytes each:
              icon_x, icon_y, arrow_x, arrow_y, flags
              flags: TASK_ICON=1, TASK_ARROW=2

Pixel fallback (pixels mode): uses simple colour-heuristic steering.
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
from mettagrid.bitworld_sprite_player import (
    AMONG_THEM_MAX_PLAYERS,
    AMONG_THEM_PHASE_GAME_OVER,
    AMONG_THEM_PHASE_LOBBY,
    AMONG_THEM_PHASE_PLAYING,
    AMONG_THEM_PHASE_ROLE_REVEAL,
    AMONG_THEM_PHASE_VOTE_RESULT,
    AMONG_THEM_PHASE_VOTING,
    SPRITE_PLAYER_BODY_FEATURE_OFFSET,
    SPRITE_PLAYER_BODY_FEATURES,
    SPRITE_PLAYER_FEATURES,
    SPRITE_PLAYER_FLAG_PLAYER_ALIVE,
    SPRITE_PLAYER_FLAG_PLAYER_GHOST,
    SPRITE_PLAYER_FLAG_PLAYER_PRESENT,
    SPRITE_PLAYER_FLAG_TASK_ARROW_VISIBLE,
    SPRITE_PLAYER_FLAG_TASK_ICON_VISIBLE,
    SPRITE_PLAYER_HEADER_FEATURES,
    SPRITE_PLAYER_KILL_ICON_INDEX,
    SPRITE_PLAYER_PLAYER_FEATURE_OFFSET,
    SPRITE_PLAYER_PLAYER_FEATURES,
    SPRITE_PLAYER_TASK_COUNT,
    SPRITE_PLAYER_TASK_FEATURE_OFFSET,
    SPRITE_PLAYER_TASK_FEATURES,
    SPRITE_PLAYER_TASKS_REMAINING_INDEX,
)
from mettagrid.policy.policy import AgentPolicy, MultiAgentPolicy
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.simulator import Action, AgentObservation

# ---------------------------------------------------------------------------
# Action index constants
# ---------------------------------------------------------------------------
_NOOP = bitworld_action_index(encode_buttons(()))
_A = bitworld_action_index(encode_buttons(("a",)))
_B = bitworld_action_index(encode_buttons(("b",)))
_UP = bitworld_action_index(encode_buttons(("up",)))
_DOWN = bitworld_action_index(encode_buttons(("down",)))
_LEFT = bitworld_action_index(encode_buttons(("left",)))
_RIGHT = bitworld_action_index(encode_buttons(("right",)))
_UP_A = bitworld_action_index(encode_buttons(("up", "a")))
_DOWN_A = bitworld_action_index(encode_buttons(("down", "a")))
_LEFT_A = bitworld_action_index(encode_buttons(("left", "a")))
_RIGHT_A = bitworld_action_index(encode_buttons(("right", "a")))
_UP_LEFT = bitworld_action_index(encode_buttons(("up", "left")))
_UP_RIGHT = bitworld_action_index(encode_buttons(("up", "right")))
_DOWN_LEFT = bitworld_action_index(encode_buttons(("down", "left")))
_DOWN_RIGHT = bitworld_action_index(encode_buttons(("down", "right")))

# Screen centre (our avatar is here in egocentric view)
_CX = SCREEN_WIDTH // 2   # 64
_CY = SCREEN_HEIGHT // 2  # 64

# Interact radius: if target is this close to centre, press A
_INTERACT_RADIUS = 18

# Spiral for exploration when no target visible
_SPIRAL = [_RIGHT, _RIGHT, _DOWN, _DOWN, _LEFT, _LEFT, _UP, _UP]
_SPIRAL_HOLD = 12  # ticks per step

# Pixel-mode task colours (PICO-8 palette)
_TASK_COLORS = frozenset([9, 10, 11, 14, 15])  # orange, yellow, bright-green, pink, peach


# ---------------------------------------------------------------------------
# Helper: direction action from (dx, dy)
# ---------------------------------------------------------------------------
def _dir_action(dx: int, dy: int, interact: bool = False) -> int:
    """Return movement action toward (dx, dy) from origin."""
    right = dx > 4
    left = dx < -4
    down = dy > 4
    up = dy < -4
    suffix = ("a",) if interact else ()
    if up and right:
        return bitworld_action_index(encode_buttons(("up", "right") + suffix))
    if up and left:
        return bitworld_action_index(encode_buttons(("up", "left") + suffix))
    if down and right:
        return bitworld_action_index(encode_buttons(("down", "right") + suffix))
    if down and left:
        return bitworld_action_index(encode_buttons(("down", "left") + suffix))
    if up:
        return bitworld_action_index(encode_buttons(("up",) + suffix))
    if down:
        return bitworld_action_index(encode_buttons(("down",) + suffix))
    if left:
        return bitworld_action_index(encode_buttons(("left",) + suffix))
    if right:
        return bitworld_action_index(encode_buttons(("right",) + suffix))
    if interact:
        return _A
    return _NOOP


# ---------------------------------------------------------------------------
# Agent sub-policy (per-agent interactive path)
# ---------------------------------------------------------------------------
class AnattaAgentPolicy(AgentPolicy):
    def __init__(self, policy_env_info: PolicyEnvInterface, parent: "AnattaPolicy", agent_id: int):
        super().__init__(policy_env_info)
        self._parent = parent
        self._agent_id = agent_id

    def step(self, obs: AgentObservation) -> Action:
        del obs
        return Action(name=bitworld_action_name(self._parent.next_action(self._agent_id)))


# ---------------------------------------------------------------------------
# Main policy
# ---------------------------------------------------------------------------
class AmongThemPolicy(MultiAgentPolicy):
    """Anatta policy — structured-observation task chasing and kill hunting.

    Supports both sprite_player (structured) and pixel (fallback) observations.
    Dispatches based on raw_observations ndim / size.
    """

    def __init__(self, policy_env_info: PolicyEnvInterface, device: str = "cpu"):
        super().__init__(policy_env_info, device=device)
        if tuple(policy_env_info.action_names) != BITWORLD_ACTION_NAMES:
            raise ValueError("AnattaPolicy requires the BitWorld AmongThem action space")
        self._tick = 0
        self._agent_steps_this_tick = 0
        # Per-agent state for voting spread
        self._agent_vote_offsets: dict[int, int] = {}

    # ------------------------------------------------------------------
    # Batch path
    # ------------------------------------------------------------------
    def step_batch(self, raw_observations: np.ndarray, raw_actions: np.ndarray) -> None:
        raw_actions[...] = self._choose_actions(raw_observations)
        self._tick += 1
        self._agent_steps_this_tick = 0

    def _choose_actions(self, raw_observations: np.ndarray) -> np.ndarray:
        batch_size = raw_observations.shape[0]
        actions = np.empty(batch_size, dtype=np.int32)

        flat = raw_observations.reshape(batch_size, -1)
        total_feats = flat.shape[1]
        is_sprite = (total_feats % SPRITE_PLAYER_FEATURES == 0) and (raw_observations.ndim <= 2)

        for i in range(batch_size):
            if is_sprite:
                # Take most recent frame if stacked
                frame_feats = flat[i, -SPRITE_PLAYER_FEATURES:]
                actions[i] = self._pick_sprite(frame_feats, agent_id=i)
            else:
                # Pixel mode: use most recent frame (last in stack)
                if raw_observations.ndim == 4:
                    frame = raw_observations[i, -1]
                else:
                    frame = raw_observations[i].reshape(SCREEN_HEIGHT, SCREEN_WIDTH)
                actions[i] = self._pick_pixel(frame, agent_id=i)

        return actions

    # ------------------------------------------------------------------
    # Sprite-player mode
    # ------------------------------------------------------------------
    def _pick_sprite(self, obs: np.ndarray, agent_id: int) -> int:
        phase = int(obs[0])
        kill_icon = int(obs[SPRITE_PLAYER_KILL_ICON_INDEX])
        tasks_remaining = int(obs[SPRITE_PLAYER_TASKS_REMAINING_INDEX])

        if phase == AMONG_THEM_PHASE_PLAYING:
            return self._play_sprite(obs, kill_icon, tasks_remaining, agent_id)
        elif phase == AMONG_THEM_PHASE_VOTING:
            return self._vote_sprite(obs, agent_id)
        elif phase in (AMONG_THEM_PHASE_ROLE_REVEAL, AMONG_THEM_PHASE_LOBBY):
            # Occasionally press A to advance / join
            return _A if self._tick % 30 < 3 else _NOOP
        else:
            # VOTE_RESULT, GAME_OVER: wait
            return _A if self._tick % 60 < 3 else _NOOP

    def _play_sprite(self, obs: np.ndarray, kill_icon: int, tasks_remaining: int, agent_id: int) -> int:
        is_imposter = kill_icon > 0

        if is_imposter and kill_icon == 255:
            # Kill ready: hunt nearest alive player
            action = self._hunt_player(obs)
            if action is not None:
                return action

        if is_imposter and kill_icon > 1:
            # On cooldown: fake task behaviour
            return self._seek_task(obs, agent_id, allow_interact=False)

        # Crewmate OR imposter-cooldown: seek and complete tasks
        return self._seek_task(obs, agent_id, allow_interact=True)

    def _seek_task(self, obs: np.ndarray, agent_id: int, allow_interact: bool) -> int:
        """Navigate toward the nearest visible task indicator and interact."""
        best_dist = 99999
        best_dx = best_dy = 0
        found = False

        for slot in range(SPRITE_PLAYER_TASK_COUNT):
            base = SPRITE_PLAYER_TASK_FEATURE_OFFSET + slot * SPRITE_PLAYER_TASK_FEATURES
            flags = int(obs[base + 4])
            if not flags:
                continue

            if flags & SPRITE_PLAYER_FLAG_TASK_ARROW_VISIBLE:
                # Arrow tells direction; treat arrow tip as target
                tx = int(obs[base + 2])
                ty = int(obs[base + 3])
            elif flags & SPRITE_PLAYER_FLAG_TASK_ICON_VISIBLE:
                tx = int(obs[base])
                ty = int(obs[base + 1])
            else:
                continue

            dx = tx - _CX
            dy = ty - _CY
            dist = dx * dx + dy * dy
            if dist < best_dist:
                best_dist = dist
                best_dx = dx
                best_dy = dy
                found = True

        if found:
            dist_sq = best_dx * best_dx + best_dy * best_dy
            if allow_interact and dist_sq <= _INTERACT_RADIUS * _INTERACT_RADIUS:
                return _A
            return _dir_action(best_dx, best_dy)

        # No task visible — exploration spiral (spread agents out with offset)
        spiral_phase = (self._tick // _SPIRAL_HOLD + agent_id * 3) % len(_SPIRAL)
        return _SPIRAL[spiral_phase]

    def _hunt_player(self, obs: np.ndarray) -> int | None:
        """Navigate toward the nearest alive non-ghost player to kill."""
        best_dist = 99999
        best_dx = best_dy = 0
        found = False

        for slot in range(AMONG_THEM_MAX_PLAYERS):
            base = SPRITE_PLAYER_PLAYER_FEATURE_OFFSET + slot * SPRITE_PLAYER_PLAYER_FEATURES
            flags = int(obs[base + 3])
            if not (flags & SPRITE_PLAYER_FLAG_PLAYER_PRESENT):
                continue
            if not (flags & SPRITE_PLAYER_FLAG_PLAYER_ALIVE):
                continue
            if flags & SPRITE_PLAYER_FLAG_PLAYER_GHOST:
                continue

            px = int(obs[base])
            py = int(obs[base + 1])
            # Skip if this is our own avatar (very close to exact centre)
            if abs(px - _CX) < 4 and abs(py - _CY) < 4:
                continue

            dx = px - _CX
            dy = py - _CY
            dist = dx * dx + dy * dy
            if dist < best_dist:
                best_dist = dist
                best_dx = dx
                best_dy = dy
                found = True

        if not found:
            return None

        dist_sq = best_dx * best_dx + best_dy * best_dy
        if dist_sq <= _INTERACT_RADIUS * _INTERACT_RADIUS:
            return _A  # kill!
        return _dir_action(best_dx, best_dy)

    def _vote_sprite(self, obs: np.ndarray, agent_id: int) -> int:
        # Simple strategy: move up to highlight skip-vote button then press A
        vote_tick = self._tick % 120
        if vote_tick < 20:
            return _UP
        elif vote_tick < 40:
            return _A
        elif vote_tick < 60:
            return _DOWN
        else:
            return _NOOP

    # ------------------------------------------------------------------
    # Pixel fallback mode
    # ------------------------------------------------------------------
    def _pick_pixel(self, frame: np.ndarray, agent_id: int) -> int:
        task_mask = np.isin(frame, list(_TASK_COLORS))
        center_count = task_mask[
            _CY - _INTERACT_RADIUS: _CY + _INTERACT_RADIUS,
            _CX - _INTERACT_RADIUS: _CX + _INTERACT_RADIUS,
        ].sum()
        if center_count > 15:
            return _A

        scores = {
            _UP: int(task_mask[: _CY - 4, :].sum()),
            _DOWN: int(task_mask[_CY + 4:, :].sum()),
            _LEFT: int(task_mask[:, : _CX - 4].sum()),
            _RIGHT: int(task_mask[:, _CX + 4:].sum()),
        }
        best_action, best_score = max(scores.items(), key=lambda kv: kv[1])
        if best_score > 8:
            return best_action

        if self._tick % 30 == 0:
            return _A

        spiral_phase = (self._tick // _SPIRAL_HOLD + agent_id * 3) % len(_SPIRAL)
        return _SPIRAL[spiral_phase]

    # ------------------------------------------------------------------
    # Per-agent interactive path
    # ------------------------------------------------------------------
    def agent_policy(self, agent_id: int) -> AgentPolicy:
        return AnattaAgentPolicy(self._policy_env_info, self, agent_id)

    def next_action(self, agent_id: int) -> int:
        self._agent_steps_this_tick += 1
        if self._agent_steps_this_tick == self._policy_env_info.num_agents:
            self._agent_steps_this_tick = 0
            self._tick += 1
        spiral_phase = (self._tick // _SPIRAL_HOLD + agent_id * 3) % len(_SPIRAL)
        return _SPIRAL[spiral_phase]
