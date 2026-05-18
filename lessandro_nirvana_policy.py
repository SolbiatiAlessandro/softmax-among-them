"""Among Them policy: lessandro-nirvana.

Buddhist theme: nirvana = liberation from suffering, the ultimate goal.
Strategy: sprite_player observation parsing for rich structured game state,
with pixel-mode fallback. Navigates to task bubbles, follows task arrows,
completes tasks by pressing A. As imposter, acts as crewmate for cover and
attacks when kill is ready. Votes during voting phase.

Observation: sprite_player mode → (SPRITE_PLAYER_FEATURES,) uint8 per agent
             pixel mode          → (4, 128, 128) uint8 per agent

Policy name: lessandro-nirvana
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
    AMONG_THEM_PHASE_PLAYING,
    AMONG_THEM_PHASE_VOTING,
    SPRITE_PLAYER_FEATURES,
    SPRITE_PLAYER_FLAG_PLAYER_ALIVE,
    SPRITE_PLAYER_FLAG_PLAYER_GHOST,
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

# ── PICO-8 palette indices used in pixel-mode fallback ────────────────────────
# Task indicators appear in vivid colours against the dark map background
TASK_COLORS = frozenset([9, 10, 11, 14, 15])  # orange, yellow, green, pink, peach

# ── Action index cache ─────────────────────────────────────────────────────────
_NOOP   = bitworld_action_index(encode_buttons(()))
_UP     = bitworld_action_index(encode_buttons(("up",)))
_DOWN   = bitworld_action_index(encode_buttons(("down",)))
_LEFT   = bitworld_action_index(encode_buttons(("left",)))
_RIGHT  = bitworld_action_index(encode_buttons(("right",)))
_A = bitworld_action_index(encode_buttons(("a",)))
_B = bitworld_action_index(encode_buttons(("b",)))

# ── Navigation constants ────────────────────────────────────────────────────────
CENTER_X = SCREEN_WIDTH // 2   # 64
CENTER_Y = SCREEN_HEIGHT // 2  # 64
# Manhattan distance threshold to trigger task interaction
INTERACT_THRESH = 22
# Distance from center below which we assume a player is ourselves
SELF_RADIUS = 14

# ── Exploration spiral (deterministic, different phase per agent) ──────────────
_SPIRAL    = [_RIGHT, _RIGHT, _DOWN, _DOWN, _LEFT, _LEFT, _UP, _UP]
SPIRAL_LEN = len(_SPIRAL)
EXPLORE_HOLD = 10  # ticks to hold each spiral direction

# ── Voting constants ──────────────────────────────────────────────────────────
VOTE_PRESS_INTERVAL = 45  # press A every N ticks during voting


def _navigate_toward(tx: int, ty: int) -> int:
    """Return directional action to move toward screen position (tx, ty)."""
    dx = tx - CENTER_X
    dy = ty - CENTER_Y
    if dx == 0 and dy == 0:
        return _A
    if abs(dx) >= abs(dy):
        return _RIGHT if dx > 0 else _LEFT
    return _DOWN if dy > 0 else _UP


# ── Per-agent wrapper ──────────────────────────────────────────────────────────

class NirvanaAgentPolicy(AgentPolicy):
    def __init__(
        self,
        policy_env_info: PolicyEnvInterface,
        parent: "LessandroNirvanaPolicy",
        agent_id: int,
    ):
        super().__init__(policy_env_info)
        self._parent = parent
        self._agent_id = agent_id

    def step(self, obs: AgentObservation) -> Action:
        del obs
        return Action(name=bitworld_action_name(self._parent.next_action(self._agent_id)))


# ── Main policy ────────────────────────────────────────────────────────────────

class LessandroNirvanaPolicy(MultiAgentPolicy):
    """Nirvana policy: rich sprite_player state → smart task & social strategy.

    sprite_player mode (1231 features per agent):
      - Reads phase, kill icon, task positions, player positions
      - Navigates directly to visible task bubbles
      - Follows task arrows when bubble is off-screen
      - Kills nearby crewmates when imposter cooldown expires
      - Votes (A press) during voting phase

    pixel mode fallback (4×128×128 per agent):
      - Colour-based task detection on the most recent frame
      - Quadrant scanning to move toward task-coloured pixels
    """

    def __init__(self, policy_env_info: PolicyEnvInterface, device: str = "cpu"):
        super().__init__(policy_env_info, device=device)
        if tuple(policy_env_info.action_names) != BITWORLD_ACTION_NAMES:
            raise ValueError(
                "LessandroNirvanaPolicy requires the BitWorld Among Them action space"
            )
        self._tick = 0
        self._agent_steps_this_tick = 0

    # ── Batch path (tournament / evaluation) ──────────────────────────────────

    def step_batch(self, raw_observations: np.ndarray, raw_actions: np.ndarray) -> None:
        raw_actions[...] = self._choose_actions(raw_observations)
        self._tick += 1
        self._agent_steps_this_tick = 0

    def _choose_actions(self, raw_observations: np.ndarray) -> np.ndarray:
        batch_size = raw_observations.shape[0]
        actions = np.empty(batch_size, dtype=np.int32)

        # Detect observation mode by shape
        is_sprite = (raw_observations.ndim == 2 and raw_observations.shape[1] == SPRITE_PLAYER_FEATURES)

        for i in range(batch_size):
            if is_sprite:
                actions[i] = self._sprite_action(raw_observations[i], agent_id=i)
            else:
                # Pixel mode: use last frame in stack
                if raw_observations.ndim == 4:
                    frame = raw_observations[i, -1]  # (H, W)
                else:
                    frame = raw_observations[i].reshape(SCREEN_HEIGHT, SCREEN_WIDTH)
                actions[i] = self._pixel_action(frame, agent_id=i)

        return actions

    # ── sprite_player action logic ─────────────────────────────────────────────

    def _sprite_action(self, obs: np.ndarray, agent_id: int) -> int:
        phase    = int(obs[0])
        kill_icon = int(obs[SPRITE_PLAYER_KILL_ICON_INDEX])
        is_imposter = kill_icon > 0
        kill_ready  = kill_icon == 255

        # ── Voting phase ──────────────────────────────────────────────────────
        if phase == AMONG_THEM_PHASE_VOTING:
            if self._tick % VOTE_PRESS_INTERVAL == 0:
                return _A
            return _NOOP

        # ── Non-playing phases (lobby, result, game over, role reveal) ────────
        if phase != AMONG_THEM_PHASE_PLAYING:
            return _NOOP

        # ── Playing phase ─────────────────────────────────────────────────────

        # Gather task observations
        task_bubbles: list[tuple[int, int]] = []
        task_arrows:  list[tuple[int, int]] = []
        for i in range(SPRITE_PLAYER_TASK_COUNT):
            base  = SPRITE_PLAYER_TASK_FEATURE_OFFSET + i * SPRITE_PLAYER_TASK_FEATURES
            flags = int(obs[base + 4])
            if flags & SPRITE_PLAYER_FLAG_TASK_ICON_VISIBLE:
                task_bubbles.append((int(obs[base]), int(obs[base + 1])))
            elif flags & SPRITE_PLAYER_FLAG_TASK_ARROW_VISIBLE:
                task_arrows.append((int(obs[base + 2]), int(obs[base + 3])))

        # ── Imposter: attack when kill ready ──────────────────────────────────
        if is_imposter and kill_ready:
            target = self._nearest_other_player(obs)
            if target is not None:
                tx, ty = target
                if abs(tx - CENTER_X) + abs(ty - CENTER_Y) < INTERACT_THRESH:
                    return _A  # kill
                return _navigate_toward(tx, ty)
            # No visible target: act like crewmate for cover
            # fall through to task logic below

        # ── Crewmate (and imposter on cooldown): complete tasks ───────────────
        if task_bubbles:
            nearest = min(
                task_bubbles,
                key=lambda p: abs(p[0] - CENTER_X) + abs(p[1] - CENTER_Y),
            )
            tx, ty = nearest
            if abs(tx - CENTER_X) + abs(ty - CENTER_Y) < INTERACT_THRESH:
                return _A  # interact / complete task
            return _navigate_toward(tx, ty)

        if task_arrows:
            # Use the arrow closest to screen edge (strongest direction hint)
            best_arrow = max(
                task_arrows,
                key=lambda p: abs(p[0] - CENTER_X) + abs(p[1] - CENTER_Y),
            )
            return _navigate_toward(best_arrow[0], best_arrow[1])

        # ── Exploration spiral (agent-staggered) ──────────────────────────────
        return _SPIRAL[(self._tick // EXPLORE_HOLD + agent_id * 3) % SPIRAL_LEN]

    def _nearest_other_player(self, obs: np.ndarray) -> tuple[int, int] | None:
        """Return screen (x, y) of the nearest alive non-self player, or None."""
        best_dist = 999
        best_pos: tuple[int, int] | None = None
        for slot in range(AMONG_THEM_MAX_PLAYERS):
            base  = SPRITE_PLAYER_PLAYER_FEATURE_OFFSET + slot * SPRITE_PLAYER_PLAYER_FEATURES
            flags = int(obs[base + 3])
            if not (flags & SPRITE_PLAYER_FLAG_PLAYER_PRESENT):
                continue
            if not (flags & SPRITE_PLAYER_FLAG_PLAYER_ALIVE):
                continue
            px, py = int(obs[base]), int(obs[base + 1])
            # Self is at screen centre; skip if within self-radius
            if abs(px - CENTER_X) < SELF_RADIUS and abs(py - CENTER_Y) < SELF_RADIUS:
                continue
            dist = abs(px - CENTER_X) + abs(py - CENTER_Y)
            if dist < best_dist:
                best_dist = dist
                best_pos = (px, py)
        return best_pos

    # ── Pixel-mode action logic (fallback) ────────────────────────────────────

    def _pixel_action(self, frame: np.ndarray, agent_id: int) -> int:
        task_mask = np.isin(frame, list(TASK_COLORS))

        center_count = task_mask[
            CENTER_Y - 12 : CENTER_Y + 12,
            CENTER_X - 12 : CENTER_X + 12,
        ].sum()
        if center_count > 15:
            return _A

        scores = {
            _UP:    int(task_mask[: CENTER_Y - 4, :].sum()),
            _DOWN:  int(task_mask[CENTER_Y + 4 :, :].sum()),
            _LEFT:  int(task_mask[:, : CENTER_X - 4].sum()),
            _RIGHT: int(task_mask[:, CENTER_X + 4 :].sum()),
        }
        best_act, best_score = max(scores.items(), key=lambda kv: kv[1])
        if best_score > 8:
            return best_act

        if self._tick % 30 == 0:
            return _A

        return _SPIRAL[(self._tick // EXPLORE_HOLD + agent_id * 3) % SPIRAL_LEN]

    # ── Per-agent path (interactive runner) ───────────────────────────────────

    def agent_policy(self, agent_id: int) -> AgentPolicy:
        return NirvanaAgentPolicy(self._policy_env_info, self, agent_id)

    def next_action(self, agent_id: int) -> int:
        """Called by NirvanaAgentPolicy — tick managed externally."""
        self._agent_steps_this_tick += 1
        if self._agent_steps_this_tick == self._policy_env_info.num_agents:
            self._agent_steps_this_tick = 0
            self._tick += 1
        return _SPIRAL[(self._tick // EXPLORE_HOLD + agent_id * 3) % SPIRAL_LEN]
