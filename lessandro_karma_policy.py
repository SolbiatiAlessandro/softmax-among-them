"""Among Them policy: lessandro-karma.

Buddhist theme: karma = intentional action and its consequences.
Strategy: task-seek as crewmate using sprite_player observation (rich game state);
hunt isolated players as imposter. Falls back to pixel heuristics when in
pixels observation mode.

Observation format (sprite_player mode):
  obs shape: (batch, SPRITE_PLAYER_FEATURES=1231), dtype uint8
  obs[0] = phase  (0=lobby, 1=playing, 2=voting, 3=vote_result, 4=game_over, 5=role_reveal)
  obs[1] = kill_icon  (0=crewmate/no kill, 1=imposter cooling, 255=imposter ready)
  obs[2] = task_progress (0-255)
  obs[3] = tasks_remaining (count)
  obs[4..4+32*32) = map grid 32x32 (PICO-8 color per tile)
  obs[1028..1092) = player slots 0-15, each 4 bytes: [x, y, color, flags]
  obs[1092..1156) = body slots 0-15, each 4 bytes: [x, y, color, 1]
  obs[1156..1231) = task slots 0-14, each 5 bytes: [icon_x, icon_y, arrow_x, arrow_y, flags]

Observation format (pixels mode, fallback):
  obs shape: (batch, 4, 128, 128), dtype uint8, PICO-8 palette indices 0-15
"""

from __future__ import annotations

import numpy as np

from mettagrid.bitworld import (
    BITWORLD_ACTION_NAMES,
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
    SPRITE_PLAYER_TASK_PROGRESS_INDEX,
    SPRITE_PLAYER_TASKS_REMAINING_INDEX,
)
from mettagrid.policy.policy import AgentPolicy, MultiAgentPolicy
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.simulator import Action, AgentObservation

# ── Action index cache ─────────────────────────────────────────────────────────
_NOOP = bitworld_action_index(encode_buttons(()))
_UP = bitworld_action_index(encode_buttons(("up",)))
_DOWN = bitworld_action_index(encode_buttons(("down",)))
_LEFT = bitworld_action_index(encode_buttons(("left",)))
_RIGHT = bitworld_action_index(encode_buttons(("right",)))
_A = bitworld_action_index(encode_buttons(("a",)))
_B = bitworld_action_index(encode_buttons(("b",)))
_UP_A = bitworld_action_index(encode_buttons(("up", "a")))
_DOWN_A = bitworld_action_index(encode_buttons(("down", "a")))
_LEFT_A = bitworld_action_index(encode_buttons(("left", "a")))
_RIGHT_A = bitworld_action_index(encode_buttons(("right", "a")))
# "select" is NOT in the trainable action set — report/vote is triggered by "b"
_REPORT = _B

# ── Screen geometry ────────────────────────────────────────────────────────────
SCREEN_CENTER = 64        # center of 128×128 screen stored as uint8
INTERACT_RADIUS = 14      # pixels — A-press interaction window
KILL_RADIUS = 16          # pixels — imposter kill window

# ── Pixel-mode heuristics (fallback) ─────────────────────────────────────────
TASK_COLORS = frozenset([9, 10, 11, 14, 15])   # orange, yellow, green, pink, peach
WALL_COLORS = frozenset([0, 1, 2, 12])          # black, dark-blue, dark-purple, void
EXPLORE_HOLD = 8
A_PRESS_INTERVAL = 25
REPORT_INTERVAL = 600
_SPIRAL = [_RIGHT, _RIGHT, _DOWN, _DOWN, _LEFT, _LEFT, _UP, _UP]


def _manhattan(ax: int, ay: int, bx: int, by: int) -> int:
    return abs(ax - bx) + abs(ay - by)


def _dir_toward(dx: int, dy: int) -> int:
    """Return the action that moves toward (dx, dy) offset."""
    if dx == 0 and dy == 0:
        return _NOOP
    if abs(dx) >= abs(dy):
        return _RIGHT if dx > 0 else _LEFT
    return _DOWN if dy > 0 else _UP


class KarmaAgentPolicy(AgentPolicy):
    def __init__(self, env_info: PolicyEnvInterface, parent: "KarmaPolicy", agent_id: int):
        super().__init__(env_info)
        self._parent = parent
        self._agent_id = agent_id

    def step(self, obs: AgentObservation) -> Action:
        del obs
        return Action(name=bitworld_action_name(self._parent.next_action(self._agent_id)))


class KarmaPolicy(MultiAgentPolicy):
    """Karma policy — smart task-seeking crewmate / hunter imposter.

    Detects observation mode automatically:
    - sprite_player (1231 features): uses exact game state
    - pixels (65536 features): falls back to colour-scanning heuristic
    """

    def __init__(self, policy_env_info: PolicyEnvInterface, device: str = "cpu"):
        super().__init__(policy_env_info, device=device)
        if tuple(policy_env_info.action_names) != BITWORLD_ACTION_NAMES:
            raise ValueError("KarmaPolicy requires the BitWorld AmongThem action space")
        self._tick = 0
        self._agent_steps_this_tick = 0
        # Per-agent exploration state
        self._explore_dir: list[int] = [0] * 64
        self._explore_counter: list[int] = [0] * 64

    # ── Batch step ────────────────────────────────────────────────────────────

    def step_batch(self, raw_observations: np.ndarray, raw_actions: np.ndarray) -> None:
        raw_actions[...] = self._choose_actions(raw_observations)
        self._tick += 1
        self._agent_steps_this_tick = 0

    def _choose_actions(self, raw_observations: np.ndarray) -> np.ndarray:
        batch_size = raw_observations.shape[0]
        actions = np.empty(batch_size, dtype=np.int32)
        flat = raw_observations.reshape(batch_size, -1)

        is_sprite = flat.shape[1] == SPRITE_PLAYER_FEATURES

        for i in range(batch_size):
            if is_sprite:
                actions[i] = self._pick_sprite(flat[i], agent_id=i)
            else:
                # Pixel fallback: use most recent frame
                if raw_observations.ndim == 4:
                    frame = raw_observations[i, -1]
                else:
                    from mettagrid.bitworld import SCREEN_HEIGHT, SCREEN_WIDTH
                    frame = raw_observations[i].reshape(SCREEN_HEIGHT, SCREEN_WIDTH)
                actions[i] = self._pick_pixel(frame, agent_id=i)

        return actions

    # ── Sprite-player logic ───────────────────────────────────────────────────

    def _pick_sprite(self, obs: np.ndarray, agent_id: int) -> int:
        phase = int(obs[0])
        kill_icon = int(obs[SPRITE_PLAYER_KILL_ICON_INDEX])
        tasks_remaining = int(obs[SPRITE_PLAYER_TASKS_REMAINING_INDEX])
        is_imposter = kill_icon > 0

        if phase == AMONG_THEM_PHASE_PLAYING:
            return self._play_sprite(obs, agent_id, is_imposter, kill_icon)
        elif phase == AMONG_THEM_PHASE_VOTING:
            return self._vote_sprite(agent_id)
        elif phase in (AMONG_THEM_PHASE_LOBBY, AMONG_THEM_PHASE_ROLE_REVEAL):
            return _NOOP if self._tick % 20 != 0 else _A
        else:
            return _NOOP

    def _play_sprite(self, obs: np.ndarray, agent_id: int, is_imposter: bool, kill_icon: int) -> int:
        if is_imposter:
            return self._imposter_sprite(obs, agent_id, kill_icon)
        return self._crewmate_sprite(obs, agent_id)

    def _crewmate_sprite(self, obs: np.ndarray, agent_id: int) -> int:
        """Navigate toward tasks and complete them."""
        # 1. Prioritise visible task icons (complete them)
        nearest_dist = 999
        nearest_action = None
        for slot in range(SPRITE_PLAYER_TASK_COUNT):
            base = SPRITE_PLAYER_TASK_FEATURE_OFFSET + slot * SPRITE_PLAYER_TASK_FEATURES
            flags = int(obs[base + 4])
            if flags & SPRITE_PLAYER_FLAG_TASK_ICON_VISIBLE:
                tx, ty = int(obs[base]), int(obs[base + 1])
                dx, dy = tx - SCREEN_CENTER, ty - SCREEN_CENTER
                dist = abs(dx) + abs(dy)
                if dist < INTERACT_RADIUS:
                    return _A  # Interact with task immediately
                if dist < nearest_dist:
                    nearest_dist = dist
                    nearest_action = _dir_toward(dx, dy)

        if nearest_action is not None:
            return nearest_action

        # 2. Follow task arrows (off-screen tasks)
        for slot in range(SPRITE_PLAYER_TASK_COUNT):
            base = SPRITE_PLAYER_TASK_FEATURE_OFFSET + slot * SPRITE_PLAYER_TASK_FEATURES
            flags = int(obs[base + 4])
            if flags & SPRITE_PLAYER_FLAG_TASK_ARROW_VISIBLE:
                ax, ay = int(obs[base + 2]), int(obs[base + 3])
                dx, dy = ax - SCREEN_CENTER, ay - SCREEN_CENTER
                return _dir_toward(dx, dy)

        # 3. Periodic A-press in case we're standing on a task
        if self._tick % A_PRESS_INTERVAL == (agent_id % A_PRESS_INTERVAL):
            return _A

        # 4. Exploration spiral
        return self._explore(agent_id)

    def _imposter_sprite(self, obs: np.ndarray, agent_id: int, kill_icon: int) -> int:
        """Hunt isolated crewmates; kill when ready."""
        # Find closest alive crewmate (non-ghost player in any slot)
        best_dist = 999
        best_dx, best_dy = 0, 0
        found = False
        for slot in range(AMONG_THEM_MAX_PLAYERS):
            base = SPRITE_PLAYER_PLAYER_FEATURE_OFFSET + slot * SPRITE_PLAYER_PLAYER_FEATURES
            flags = int(obs[base + 3])
            if not (flags & SPRITE_PLAYER_FLAG_PLAYER_PRESENT):
                continue
            if not (flags & SPRITE_PLAYER_FLAG_PLAYER_ALIVE):
                continue
            # slot 0 may or may not be self — check color matches kill cooldown context
            px, py = int(obs[base]), int(obs[base + 1])
            dx, dy = px - SCREEN_CENTER, py - SCREEN_CENTER
            dist = abs(dx) + abs(dy)
            # Skip players that are at exactly screen center (might be self)
            if dist < 3:
                continue
            if dist < best_dist:
                best_dist = dist
                best_dx, best_dy = dx, dy
                found = True

        if found:
            if best_dist < KILL_RADIUS and kill_icon == 255:
                return _A  # Kill!
            return _dir_toward(best_dx, best_dy)

        # No targets visible — explore
        return self._explore(agent_id)

    def _vote_sprite(self, agent_id: int) -> int:
        """Vote or skip during voting phase."""
        if self._tick % 3 == 0:
            return _A   # Vote / confirm
        return _NOOP

    # ── Pixel-mode fallback ───────────────────────────────────────────────────

    def _pick_pixel(self, frame: np.ndarray, agent_id: int) -> int:
        from mettagrid.bitworld import SCREEN_HEIGHT, SCREEN_WIDTH
        CENTER_R = SCREEN_HEIGHT // 2
        CENTER_C = SCREEN_WIDTH // 2
        HALF = 12

        task_mask = np.isin(frame, list(TASK_COLORS))

        # 1. Interact if task pixels near center
        center_count = task_mask[CENTER_R - HALF: CENTER_R + HALF, CENTER_C - HALF: CENTER_C + HALF].sum()
        if center_count > 15:
            return _A

        # 2. Periodic B-press to report bodies / advance voting UI
        if (self._tick + agent_id * 37) % REPORT_INTERVAL < 3:
            return _REPORT

        # 3. Move toward task-rich quadrant
        quadrant_scores = {
            _UP: int(task_mask[:CENTER_R - 4, :].sum()),
            _DOWN: int(task_mask[CENTER_R + 4:, :].sum()),
            _LEFT: int(task_mask[:, :CENTER_C - 4].sum()),
            _RIGHT: int(task_mask[:, CENTER_C + 4:].sum()),
        }
        best_action, best_score = max(quadrant_scores.items(), key=lambda kv: kv[1])
        if best_score > 8:
            return best_action

        # 4. Periodic blind A-press
        if self._tick % A_PRESS_INTERVAL == 0:
            return _A

        # 5. Spiral exploration
        return self._explore(agent_id)

    # ── Exploration helper ────────────────────────────────────────────────────

    def _explore(self, agent_id: int) -> int:
        """Return next spiral direction for this agent."""
        counter = self._explore_counter[agent_id]
        dir_idx = self._explore_dir[agent_id]
        if counter <= 0:
            dir_idx = (dir_idx + 1) % len(_SPIRAL)
            self._explore_dir[agent_id] = dir_idx
            self._explore_counter[agent_id] = EXPLORE_HOLD
        else:
            self._explore_counter[agent_id] = counter - 1
        return _SPIRAL[dir_idx]

    # ── Per-agent path ────────────────────────────────────────────────────────

    def agent_policy(self, agent_id: int) -> AgentPolicy:
        return KarmaAgentPolicy(self._policy_env_info, self, agent_id)

    def next_action(self, agent_id: int) -> int:
        self._agent_steps_this_tick += 1
        if self._agent_steps_this_tick == self._policy_env_info.num_agents:
            self._agent_steps_this_tick = 0
            self._tick += 1
        spiral_index = (self._tick // EXPLORE_HOLD + agent_id * 3) % len(_SPIRAL)
        return _SPIRAL[spiral_index]


# Required alias for tournament runner
AmongThemPolicy = KarmaPolicy
