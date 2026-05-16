"""Among Them policy: lessandro-nirvana.

Buddhist theme: nirvana = liberation, the end of suffering and wandering.
Strategy: structured task navigation using sprite_player observations.

Handles BOTH observation modes:
  - sprite_player (flat 1231-dim array): full game state, preferred
  - pixel (4 × 128 × 128 array): color-based fallback

Sprite_player observation layout:
  obs[0]          : game phase (0=LOBBY 1=PLAYING 2=VOTING 3=VOTE_RESULT 4=GAME_OVER 5=ROLE)
  obs[1]          : kill_icon  (0=no-kill 1=cooldown 255=ready-to-kill → imposter)
  obs[2]          : task progress byte (0-255, scales with % completion)
  obs[3]          : tasks remaining counter
  obs[4:1028]     : 32×32 map grid (PICO-8 color indices, camera-centered)
  obs[1028:1092]  : 16 players × 4 bytes  [x, y, color, flags]
  obs[1092:1156]  : 16 bodies  × 4 bytes  [x, y, color, present]
  obs[1156:1231]  : 15 tasks   × 5 bytes  [icon_x, icon_y, arrow_x, arrow_y, flags]

Player flags: present=1, alive=4, flip_h=16, ghost=32
Task flags:   icon_visible=1, arrow_visible=2

Scoring: crewmates win by completing all tasks; imposters win by killing.
Policy detects imposter role via obs[1] > 0 and switches strategies.
"""

from __future__ import annotations

import numpy as np

from mettagrid.bitworld import (
    BITWORLD_ACTION_NAMES,
    bitworld_action_index,
    bitworld_action_name,
    encode_buttons,
)
from mettagrid.policy.policy import AgentPolicy, MultiAgentPolicy
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.simulator import Action, AgentObservation

# ---------------------------------------------------------------------------
# Observation constants (must match mettagrid.bitworld_sprite_player)
# ---------------------------------------------------------------------------
_SP_FEATURES = 1231
_SP_PLAYER_OFF = 1028   # obs[1028 + slot*4] = player x
_SP_BODY_OFF = 1092
_SP_TASK_OFF = 1156
_SP_PLAYER_F = 4        # features per player
_SP_BODY_F = 4          # features per body
_SP_TASK_F = 5          # features per task
_SP_TASK_N = 15
_SP_PLAYER_N = 16

_PHASE_LOBBY = 0
_PHASE_PLAYING = 1
_PHASE_VOTING = 2
_PHASE_VOTE_RESULT = 3
_PHASE_GAME_OVER = 4
_PHASE_ROLE = 5

_FLAG_PRESENT = 1
_FLAG_ALIVE = 4
_FLAG_TASK_ICON = 1
_FLAG_TASK_ARROW = 2

_KILL_READY = 255
_CENTER = 64        # camera keeps player at screen center (128×128)
_INTERACT_R = 13    # interact radius (pixels from center)

# ---------------------------------------------------------------------------
# Pre-computed action indices
# ---------------------------------------------------------------------------
_NOOP = bitworld_action_index(encode_buttons(()))
_A = bitworld_action_index(encode_buttons(("a",)))
_UP = bitworld_action_index(encode_buttons(("up",)))
_UP_A = bitworld_action_index(encode_buttons(("up", "a")))
_DOWN = bitworld_action_index(encode_buttons(("down",)))
_DOWN_A = bitworld_action_index(encode_buttons(("down", "a")))
_LEFT = bitworld_action_index(encode_buttons(("left",)))
_LEFT_A = bitworld_action_index(encode_buttons(("left", "a")))
_RIGHT = bitworld_action_index(encode_buttons(("right",)))
_RIGHT_A = bitworld_action_index(encode_buttons(("right", "a")))
_UL = bitworld_action_index(encode_buttons(("up", "left")))
_UL_A = bitworld_action_index(encode_buttons(("up", "left", "a")))
_UR = bitworld_action_index(encode_buttons(("up", "right")))
_UR_A = bitworld_action_index(encode_buttons(("up", "right", "a")))
_DL = bitworld_action_index(encode_buttons(("down", "left")))
_DL_A = bitworld_action_index(encode_buttons(("down", "left", "a")))
_DR = bitworld_action_index(encode_buttons(("down", "right")))
_DR_A = bitworld_action_index(encode_buttons(("down", "right", "a")))

# Exploration pattern: 4-quadrant sweep to find tasks
_EXPLORE = (
    [_RIGHT] * 20 + [_DOWN] * 20 + [_LEFT] * 40 + [_UP] * 40
    + [_RIGHT] * 40 + [_DOWN] * 20
)

# PICO-8 bright "task indicator" colors (orange, yellow, green, pink, peach)
_TASK_COLORS = frozenset([9, 10, 11, 14, 15])


class AmongThemAgentPolicy(AgentPolicy):
    def __init__(self, policy_env_info: PolicyEnvInterface, parent: "AmongThemPolicy", agent_id: int):
        super().__init__(policy_env_info)
        self._parent = parent
        self._agent_id = agent_id

    def step(self, obs: AgentObservation) -> Action:
        del obs
        return Action(name=bitworld_action_name(self._parent.next_action(self._agent_id)))


class AmongThemPolicy(MultiAgentPolicy):
    """Nirvana policy — sprite_player task-seeking with imposter awareness."""

    def __init__(self, policy_env_info: PolicyEnvInterface, device: str = "cpu"):
        super().__init__(policy_env_info, device=device)
        if tuple(policy_env_info.action_names) != BITWORLD_ACTION_NAMES:
            raise ValueError("AmongThemPolicy requires the BitWorld Among Them action space")
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
        batch_size = raw_observations.shape[0]
        flat = raw_observations.reshape(batch_size, -1)
        is_sprite = flat.shape[1] == _SP_FEATURES
        actions = np.empty(batch_size, dtype=np.int32)
        for i in range(batch_size):
            if is_sprite:
                actions[i] = _pick_sprite(flat[i], self._tick, i)
            else:
                actions[i] = _pick_pixel(flat[i], self._tick, i)
        return actions

    # ------------------------------------------------------------------
    # Per-agent path (interactive runner)
    # ------------------------------------------------------------------

    def agent_policy(self, agent_id: int) -> AgentPolicy:
        return AmongThemAgentPolicy(self._policy_env_info, self, agent_id)

    def next_action(self, agent_id: int) -> int:
        self._agent_steps_this_tick += 1
        if self._agent_steps_this_tick == self._policy_env_info.num_agents:
            self._agent_steps_this_tick = 0
            self._tick += 1
        explore_idx = (self._tick // 1 + agent_id * 17) % len(_EXPLORE)
        return _EXPLORE[explore_idx]


# ---------------------------------------------------------------------------
# Sprite-player action logic
# ---------------------------------------------------------------------------

def _navigate(dx: int, dy: int, use_a: bool) -> int:
    """Return best action toward (dx, dy) offset from screen center."""
    adx, ady = abs(dx), abs(dy)
    diag = adx > 8 and ady > 8
    if diag:
        if dx > 0 and dy < 0:
            return _UR_A if use_a else _UR
        if dx < 0 and dy < 0:
            return _UL_A if use_a else _UL
        if dx > 0 and dy > 0:
            return _DR_A if use_a else _DR
        return _DL_A if use_a else _DL
    if adx >= ady:
        return (_RIGHT_A if use_a else _RIGHT) if dx > 0 else (_LEFT_A if use_a else _LEFT)
    return (_DOWN_A if use_a else _DOWN) if dy > 0 else (_UP_A if use_a else _UP)


def _pick_sprite(obs: np.ndarray, tick: int, slot: int) -> int:
    phase = int(obs[0])
    kill_icon = int(obs[1])

    # Non-playing phases: advance UI quickly
    if phase != _PHASE_PLAYING:
        if phase == _PHASE_VOTING:
            # Vote every 25 ticks, otherwise noop
            return _A if tick % 25 < 4 else _NOOP
        return _A if tick % 15 < 3 else _NOOP

    # ---- PLAYING PHASE ----

    # 1. Find nearest task icon (preferred)
    best_tx: int | None = None
    best_ty: int | None = None
    best_dist = 10**9

    for t in range(_SP_TASK_N):
        base = _SP_TASK_OFF + t * _SP_TASK_F
        flags = int(obs[base + 4])
        if flags & _FLAG_TASK_ICON:
            tx, ty = int(obs[base]), int(obs[base + 1])
            d = (tx - _CENTER) ** 2 + (ty - _CENTER) ** 2
            if d < best_dist:
                best_dist = d
                best_tx, best_ty = tx, ty

    # 2. Fall back to task arrows
    if best_tx is None:
        for t in range(_SP_TASK_N):
            base = _SP_TASK_OFF + t * _SP_TASK_F
            flags = int(obs[base + 4])
            if flags & _FLAG_TASK_ARROW:
                tx, ty = int(obs[base + 2]), int(obs[base + 3])
                d = (tx - _CENTER) ** 2 + (ty - _CENTER) ** 2 + 10000  # deprioritize vs icons
                if d < best_dist:
                    best_dist = d
                    best_tx, best_ty = tx, ty

    # 3. Imposter kill: find nearest alive player when kill is ready
    if kill_icon == _KILL_READY and best_tx is None:
        for p in range(_SP_PLAYER_N):
            base = _SP_PLAYER_OFF + p * _SP_PLAYER_F
            flags = int(obs[base + 3])
            if not (flags & _FLAG_PRESENT) or not (flags & _FLAG_ALIVE):
                continue
            px, py = int(obs[base]), int(obs[base + 1])
            if abs(px - _CENTER) < 6 and abs(py - (_CENTER + 1)) < 6:
                continue  # skip self (player at screen center)
            d = (px - _CENTER) ** 2 + (py - _CENTER) ** 2
            if d < best_dist:
                best_dist = d
                best_tx, best_ty = px, py

    # 4. Navigate to target
    if best_tx is not None:
        dx = best_tx - _CENTER
        dy = best_ty - _CENTER
        if abs(dx) <= _INTERACT_R and abs(dy) <= _INTERACT_R:
            return _A  # at target: interact / kill
        use_a = (tick % 8 == 0)
        return _navigate(dx, dy, use_a)

    # 5. Exploration pattern (slot offset so agents spread out)
    explore_idx = (tick // 2 + slot * 41) % len(_EXPLORE)
    return _EXPLORE[explore_idx]


# ---------------------------------------------------------------------------
# Pixel fallback action logic (same as dharma, improved thresholds)
# ---------------------------------------------------------------------------

_TASK_COLOR_LIST = list(_TASK_COLORS)
_SPIRAL = [_RIGHT, _RIGHT, _DOWN, _DOWN, _LEFT, _LEFT, _UP, _UP]
_EXPLORE_HOLD = 12
_A_INTERVAL = 28


def _pick_pixel(flat_obs: np.ndarray, tick: int, slot: int) -> int:
    """Pixel-mode fallback: color-based task navigation."""
    h, w = 128, 128
    try:
        frame = flat_obs[-h * w:].reshape(h, w)
    except Exception:
        return _NOOP

    task_mask = np.isin(frame, _TASK_COLOR_LIST)
    half = 12
    c = 64
    center_count = int(task_mask[c - half : c + half, c - half : c + half].sum())
    if center_count > 12:
        return _A

    up_s = int(task_mask[: c - 4, :].sum())
    down_s = int(task_mask[c + 4 :, :].sum())
    left_s = int(task_mask[:, : c - 4].sum())
    right_s = int(task_mask[:, c + 4 :].sum())
    scores = {_UP: up_s, _DOWN: down_s, _LEFT: left_s, _RIGHT: right_s}
    best_act, best_sc = max(scores.items(), key=lambda kv: kv[1])
    if best_sc > 6:
        return best_act

    if tick % _A_INTERVAL == 0:
        return _A

    si = (tick // _EXPLORE_HOLD + slot * 3) % len(_SPIRAL)
    return _SPIRAL[si]
