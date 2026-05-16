"""BitWorld WebSocket player for Among Them (coworld Observatory).

Reads COGAMES_ENGINE_WS_URL and plays using the lessandro-nirvana policy.

URL format: ws://<host>:<port>/player?slot=<N>&token=<TOKEN>
  or:       ws://<host>:<port>/sprite_player?slot=<N>&token=<TOKEN>

The path determines the observation mode:
  /player        → raw pixel mode (8192 bytes/frame, packed PICO-8)
  /sprite_player → structured sprite mode (SpritePlayerObservationAdapter)
"""

from __future__ import annotations

import asyncio
import os
import sys
from urllib.parse import parse_qs, urlparse

import numpy as np
import websockets

sys.path.insert(0, "/app")
from lessandro_nirvana_policy import (
    _EXPLORE,
    _NOOP,
    _pick_pixel,
    _pick_sprite,
    _SP_FEATURES,
)

PIXEL_FRAME_BYTES = 8192   # 128 × 128 / 2 (4-bit packed)
PIXEL_H = 128
PIXEL_W = 128
FRAME_STACK = 4

PACKET_INPUT = 0x00
PACKET_SPRITE_INPUT = 0x84


def _slot_from_url(url: str) -> int:
    parsed = urlparse(url)
    qs = parse_qs(parsed.query)
    slot_vals = qs.get("slot", ["0"])
    try:
        return int(slot_vals[0])
    except (ValueError, IndexError):
        return 0


def _unpack_frame(packed: bytes) -> np.ndarray:
    raw = np.frombuffer(packed, dtype=np.uint8)
    frame = np.empty(PIXEL_H * PIXEL_W, dtype=np.uint8)
    frame[0::2] = raw & 0x0F
    frame[1::2] = raw >> 4
    return frame.reshape(PIXEL_H, PIXEL_W)


async def _pixel_loop(ws: websockets.WebSocketClientProtocol, slot: int) -> None:
    from mettagrid.bitworld import BITWORLD_ACTION_MASKS

    frame_stack: list[np.ndarray] = []
    tick = 0

    async for message in ws:
        if not isinstance(message, bytes):
            continue
        if len(message) != PIXEL_FRAME_BYTES:
            continue

        frame = _unpack_frame(message)
        frame_stack.append(frame)
        if len(frame_stack) > FRAME_STACK:
            frame_stack.pop(0)

        if len(frame_stack) == FRAME_STACK:
            obs_stack = np.stack(frame_stack)   # (4, 128, 128)
            flat = obs_stack.reshape(-1)
            action_idx = _pick_pixel(flat, tick, slot)
        else:
            explore_idx = (tick // 2 + slot * 41) % len(_EXPLORE)
            action_idx = _EXPLORE[explore_idx]

        mask = int(BITWORLD_ACTION_MASKS[action_idx])
        await ws.send(bytes([PACKET_INPUT, mask]))
        tick += 1


async def _sprite_loop(ws: websockets.WebSocketClientProtocol, slot: int) -> None:
    from mettagrid import bitworld_sprite_player as sp
    from mettagrid.bitworld import BITWORLD_ACTION_MASKS

    adapter = sp.SpritePlayerObservationAdapter()
    tick = 0

    async for message in ws:
        if not isinstance(message, bytes):
            continue

        changed = adapter.apply_packet(message)
        if not changed:
            continue

        obs = adapter.observation()
        if obs.shape[0] != _SP_FEATURES:
            continue

        action_idx = _pick_sprite(obs, tick, slot)
        mask = int(BITWORLD_ACTION_MASKS[action_idx])
        await ws.send(bytes([PACKET_SPRITE_INPUT, mask]))
        tick += 1


async def main() -> None:
    url = os.environ["COGAMES_ENGINE_WS_URL"]
    slot = _slot_from_url(url)
    is_sprite = "/sprite_player" in url

    print(f"[nirvana] connecting slot={slot} mode={'sprite' if is_sprite else 'pixel'} url={url}")
    sys.stdout.flush()

    try:
        async with websockets.connect(url, ping_interval=None) as ws:
            if is_sprite:
                await _sprite_loop(ws, slot)
            else:
                await _pixel_loop(ws, slot)
    except websockets.exceptions.ConnectionClosed:
        pass
    print(f"[nirvana] episode ended after slot={slot}")


if __name__ == "__main__":
    asyncio.run(main())
