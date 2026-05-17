"""Among Them sprite_player policy: lessandro-nirvana.

Buddhist theme: nirvana = liberation from the cycle of suffering.
Uses the /sprite_player observation protocol for structured game state:
  - Exact task icon / task arrow positions on screen
  - All player positions and alive/dead status
  - Kill-icon flag → know we are imposter with kill ready
  - Game phase: playing / voting / vote-result / role-reveal / game-over

Observation layout (SPRITE_PLAYER_FEATURES = 1231 bytes per frame):
  obs[0]          phase
  obs[1]          kill_icon  (0=no, 1=cooldown, 255=kill ready)
  obs[2]          task progress bar byte
  obs[3]          tasks-remaining counter
  obs[4:1028]     32×32 map grid (camera-relative tile colors)
  obs[1028:1092]  16 player slots × 4 bytes (x+1, y+1, color, flags)
  obs[1092:1156]  16 body slots   × 4 bytes (x+1, y+1, color, present)
  obs[1156:1231]  15 task  slots  × 5 bytes (icon_x, icon_y, arrow_x, arrow_y, flags)
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym

from mettagrid.bitworld import (
    BITWORLD_ACTION_COUNT,
    BITWORLD_ACTION_NAMES,
    bitworld_action_index,
    bitworld_action_name,
    encode_buttons,
)
from mettagrid.bitworld_sprite_player import (
    AMONG_THEM_MAX_PLAYERS,
    AMONG_THEM_PHASE_GAME_OVER,
    AMONG_THEM_PHASE_PLAYING,
    AMONG_THEM_PHASE_ROLE_REVEAL,
    AMONG_THEM_PHASE_VOTE_RESULT,
    AMONG_THEM_PHASE_VOTING,
    SPRITE_PLAYER_BODY_FEATURE_OFFSET,
    SPRITE_PLAYER_BODY_FEATURES,
    SPRITE_PLAYER_FEATURES,
    SPRITE_PLAYER_FLAG_PLAYER_ALIVE,
    SPRITE_PLAYER_FLAG_PLAYER_PRESENT,
    SPRITE_PLAYER_FLAG_TASK_ARROW_VISIBLE,
    SPRITE_PLAYER_FLAG_TASK_ICON_VISIBLE,
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

# ── Action shortcuts ──────────────────────────────────────────────────────────
_NOOP   = bitworld_action_index(encode_buttons(()))
_UP     = bitworld_action_index(encode_buttons(("up",)))
_DOWN   = bitworld_action_index(encode_buttons(("down",)))
_LEFT   = bitworld_action_index(encode_buttons(("left",)))
_RIGHT  = bitworld_action_index(encode_buttons(("right",)))
_A      = bitworld_action_index(encode_buttons(("a",)))
_B      = bitworld_action_index(encode_buttons(("b",)))
# NOTE: "select" is not in the trainable action set (only up/down/left/right/a/b)
# Use A to vote/confirm and B to skip/report during voting phase.

# Direction + A (move while interacting)
_UP_A    = bitworld_action_index(encode_buttons(("up", "a")))
_DOWN_A  = bitworld_action_index(encode_buttons(("down", "a")))
_LEFT_A  = bitworld_action_index(encode_buttons(("left", "a")))
_RIGHT_A = bitworld_action_index(encode_buttons(("right", "a")))

FRAME_STACK = 4
SCREEN_CENTER = 64        # camera keeps player roughly centred on 128×128 screen

CLOSE_INTERACT_DIST = 10  # pixels: press A every tick
NEAR_INTERACT_DIST  = 22  # pixels: press A every other tick + move
REPORT_DIST         = 22  # pixels: report nearby body
KILL_DIST           = 22  # pixels: press A to kill

_EXPLORE = [_RIGHT, _RIGHT, _RIGHT, _DOWN, _DOWN, _DOWN,
            _LEFT,  _LEFT,  _LEFT,  _UP,   _UP,   _UP]
EXPLORE_HOLD = 6  # ticks per exploration direction step


class NirvanaAgentPolicy(AgentPolicy):
    """Per-agent wrapper for the interactive (non-batch) runner path."""

    def __init__(self, env_info: PolicyEnvInterface, parent: "NirvanaPolicy", agent_id: int):
        super().__init__(env_info)
        self._parent   = parent
        self._agent_id = agent_id
        self._tick     = 0

    def step(self, obs: AgentObservation) -> Action:
        del obs
        action = self._parent.next_action(self._agent_id, self._tick)
        self._tick += 1
        return Action(name=bitworld_action_name(action))


class NirvanaPolicy(MultiAgentPolicy):
    """Sprite-player-based Among Them policy.

    Declares observation_kind='sprite_player' via an overridden
    policy_env_info so the runner connects to /sprite_player and
    delivers structured game state instead of raw pixels.
    """

    _SPRITE_ENV_INFO: PolicyEnvInterface | None = None

    def __init__(self, policy_env_info: PolicyEnvInterface, device: str = "cpu"):
        super().__init__(policy_env_info, device=device)

        # Build (and cache) a sprite_player env interface.
        # The runner reads policy.policy_env_info to decide which WebSocket
        # path to connect to; returning sprite_player here causes /sprite_player.
        if NirvanaPolicy._SPRITE_ENV_INFO is None:
            obs_space = gym.spaces.Box(
                low=0, high=255,
                shape=(FRAME_STACK * SPRITE_PLAYER_FEATURES,),
                dtype=np.uint8,
            )
            act_space = gym.spaces.Discrete(BITWORLD_ACTION_COUNT)
            NirvanaPolicy._SPRITE_ENV_INFO = PolicyEnvInterface.from_spaces(
                observation_space=obs_space,
                action_space=act_space,
                num_agents=policy_env_info.num_agents,
                action_names=list(BITWORLD_ACTION_NAMES),
                observation_kind="sprite_player",
            )
        self._sprite_env_info = NirvanaPolicy._SPRITE_ENV_INFO
        self._tick = 0

    # ── Override so the runner sees sprite_player ──────────────────────────
    @property
    def policy_env_info(self) -> PolicyEnvInterface:
        return self._sprite_env_info

    # ── Per-agent path (interactive runner) ───────────────────────────────
    def agent_policy(self, agent_id: int) -> AgentPolicy:
        return NirvanaAgentPolicy(self._sprite_env_info, self, agent_id)

    def next_action(self, agent_id: int, tick: int) -> int:
        return _EXPLORE[(tick // EXPLORE_HOLD + agent_id * 5) % len(_EXPLORE)]

    # ── Batch path (tournament / evaluation) ──────────────────────────────
    def step_batch(self, raw_observations: np.ndarray, raw_actions: np.ndarray) -> None:
        batch_size = raw_observations.shape[0]

        if raw_observations.ndim == 2 and raw_observations.shape[1] >= SPRITE_PLAYER_FEATURES:
            # sprite_player: (batch, frame_stack * features) → take last frame
            last_frame = raw_observations[:, -SPRITE_PLAYER_FEATURES:]
        elif raw_observations.ndim == 4:
            # Pixel fallback: just explore
            for i in range(batch_size):
                raw_actions[i] = _EXPLORE[(self._tick // EXPLORE_HOLD + i * 5) % len(_EXPLORE)]
            self._tick += 1
            return
        else:
            for i in range(batch_size):
                raw_actions[i] = _NOOP
            self._tick += 1
            return

        for i in range(batch_size):
            raw_actions[i] = self._pick_action(last_frame[i], agent_id=i)
        self._tick += 1

    # ── Per-agent decision ────────────────────────────────────────────────
    def _pick_action(self, obs: np.ndarray, agent_id: int) -> int:
        phase     = int(obs[0])
        kill_icon = int(obs[SPRITE_PLAYER_KILL_ICON_INDEX])

        # ─ Non-playing phases ─────────────────────────────────────────────
        if phase == AMONG_THEM_PHASE_VOTING:
            # Navigate vote panel with dpad and confirm with A; B to skip
            t = self._tick + agent_id * 13
            if t % 30 < 3:
                return _B               # skip / report
            if t % 15 < 3:
                return _A               # confirm / vote
            if t % 15 < 6:
                return _DOWN            # navigate to next candidate
            return _NOOP

        if phase in (AMONG_THEM_PHASE_VOTE_RESULT,
                     AMONG_THEM_PHASE_ROLE_REVEAL,
                     AMONG_THEM_PHASE_GAME_OVER):
            return _A if self._tick % 8 == 0 else _NOOP

        if phase != AMONG_THEM_PHASE_PLAYING:
            return _NOOP   # LOBBY → wait

        # ─ Playing phase ──────────────────────────────────────────────────

        # IMPOSTER with kill ready: find and eliminate nearest player
        if kill_icon == 255:
            action = self._imposter_kill(obs, agent_id)
            if action is not None:
                return action

        # CREWMATE (or imposter on cooldown): complete nearest task
        action = self._navigate_task(obs, agent_id)
        if action is not None:
            return action

        # Report any visible body
        action = self._report_body(obs)
        if action is not None:
            return action

        # No task visible → explore systematically
        return self._explore(agent_id)

    # ── Helpers ───────────────────────────────────────────────────────────

    def _navigate_task(self, obs: np.ndarray, agent_id: int) -> int | None:
        """Move toward the nearest visible task icon; follow task arrow if none."""
        best_dist = 9999
        best_dx   = 0
        best_dy   = 0
        has_icon  = False
        arrow_dx  = 0
        arrow_dy  = 0
        has_arrow = False

        for slot in range(SPRITE_PLAYER_TASK_COUNT):
            base  = SPRITE_PLAYER_TASK_FEATURE_OFFSET + slot * SPRITE_PLAYER_TASK_FEATURES
            flags = int(obs[base + 4])

            if flags & SPRITE_PLAYER_FLAG_TASK_ICON_VISIBLE:
                tx   = int(obs[base])
                ty   = int(obs[base + 1])
                dx   = tx - SCREEN_CENTER
                dy   = ty - SCREEN_CENTER
                dist = abs(dx) + abs(dy)
                if dist < best_dist:
                    best_dist = dist
                    best_dx   = dx
                    best_dy   = dy
                    has_icon  = True

            elif flags & SPRITE_PLAYER_FLAG_TASK_ARROW_VISIBLE:
                arrow_dx = int(obs[base + 2]) - SCREEN_CENTER
                arrow_dy = int(obs[base + 3]) - SCREEN_CENTER
                has_arrow = True

        if has_icon:
            if best_dist < CLOSE_INTERACT_DIST:
                return _A
            if best_dist < NEAR_INTERACT_DIST:
                # Interleave move+A with pure A
                return self._dir_to_a(best_dx, best_dy) if self._tick % 2 == 0 else _A
            return self._dir_to(best_dx, best_dy)

        if has_arrow:
            # Follow arrow toward off-screen task
            return self._dir_to(arrow_dx, arrow_dy)

        return None

    def _imposter_kill(self, obs: np.ndarray, agent_id: int) -> int | None:
        """Locate nearest visible alive player and kill."""
        nearest_dist = 9999
        nearest_dx   = 0
        nearest_dy   = 0

        for slot in range(AMONG_THEM_MAX_PLAYERS):
            base  = SPRITE_PLAYER_PLAYER_FEATURE_OFFSET + slot * SPRITE_PLAYER_PLAYER_FEATURES
            flags = int(obs[base + 3])
            if not (flags & SPRITE_PLAYER_FLAG_PLAYER_PRESENT):
                continue
            if not (flags & SPRITE_PLAYER_FLAG_PLAYER_ALIVE):
                continue
            px = int(obs[base])     - 1   # stored as x+1
            py = int(obs[base + 1]) - 1
            if px <= 0 and py <= 0:
                continue
            dx   = px - SCREEN_CENTER
            dy   = py - SCREEN_CENTER
            dist = abs(dx) + abs(dy)
            if dist < 6:
                continue   # probably our own character at screen center
            if dist < nearest_dist:
                nearest_dist = dist
                nearest_dx   = dx
                nearest_dy   = dy

        if nearest_dist <= KILL_DIST:
            return _A
        if nearest_dist < 9999:
            return self._dir_to(nearest_dx, nearest_dy)
        return None

    def _report_body(self, obs: np.ndarray) -> int | None:
        """Report a body if one is close to screen center."""
        for slot in range(AMONG_THEM_MAX_PLAYERS):
            base    = SPRITE_PLAYER_BODY_FEATURE_OFFSET + slot * SPRITE_PLAYER_BODY_FEATURES
            present = int(obs[base + 3])
            if not present:
                continue
            bx = int(obs[base])     - 1
            by = int(obs[base + 1]) - 1
            if bx <= 0 and by <= 0:
                continue
            dx   = bx - SCREEN_CENTER
            dy   = by - SCREEN_CENTER
            dist = abs(dx) + abs(dy)
            if dist < REPORT_DIST:
                return _B
        return None

    def _dir_to(self, dx: int, dy: int) -> int:
        """Cardinal direction toward (dx, dy)."""
        if abs(dx) >= abs(dy):
            return _RIGHT if dx > 0 else _LEFT
        return _DOWN if dy > 0 else _UP

    def _dir_to_a(self, dx: int, dy: int) -> int:
        """Cardinal direction + A (move and interact simultaneously)."""
        if abs(dx) >= abs(dy):
            return _RIGHT_A if dx > 0 else _LEFT_A
        return _DOWN_A if dy > 0 else _UP_A

    def _explore(self, agent_id: int) -> int:
        return _EXPLORE[(self._tick // EXPLORE_HOLD + agent_id * 5) % len(_EXPLORE)]
