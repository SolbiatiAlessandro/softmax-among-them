# lessandro-nirvana — LLM-Enhanced Among Them Policy

Buddhist theme: nirvana = enlightened state, liberation from suffering.

Based on `italkalot.nim` from the bitworld repository — the most advanced
Among Them bot, now enhanced to use OpenRouter (Claude) for voting decisions.

## Strategy

### Crewmate
1. A* pathfinding to task stations
2. Detect task icons, stand still and hold A to complete
3. Report dead bodies immediately
4. Voting: read chat for "sus" accusations, vote chat suspect first, then
   most recently seen player. With LLM enabled: Claude decides based on
   transcript evidence.

### Imposter
1. Navigate to "fake targets" (task-like locations) to appear to be working
2. Only kill when a lone crewmate is visible and kill cooldown is ready
3. Flee from visible dead bodies (to avoid being near the report)
4. Voting: accuse a random crewmate with fake "sus" chat. With LLM: Claude
   generates a contextually appropriate accusation.

### LLM Integration
When `OPENROUTER_API_KEY` is set (at container runtime), voting is enhanced
with Claude claude-haiku-4.5 via OpenRouter:
- **Chat**: AI generates short, context-aware vote chat messages
- **Vote**: AI reads the full vote transcript and picks the best target

Without the API key, falls back to heuristic voting (still good).

## Build

```bash
docker build --platform=linux/amd64 -t lessandro-nirvana:latest .
```

Requires internet access to download Alpine packages and the Nim compiler.

## Test (after downloading the coworld package)

```bash
# First download the Among Them coworld package
uv run coworld download among_them

# Run a certification smoke episode
uv run coworld run-episode ./coworld/<coworld-id>/coworld_manifest.json \
  lessandro-nirvana:latest --timeout-seconds 120
```

## Submit

```bash
# Set environment (fresh auth token required)
uv run softmax set-token '<TOKEN_FROM_BROWSER>'

# Certify and submit to Among Them Daily league
uv run coworld certify \
  --league <LEAGUE_ID> \
  --image lessandro-nirvana:latest \
  --name lessandro-nirvana \
  --env OPENROUTER_API_KEY=<YOUR_KEY>
```

## Files
- `lessandro_nirvana.nim` — main Nim policy (based on italkalot.nim)
- `openrouter.nim` — OpenRouter API adapter (replaces bitworld/ais/openai.nim)
- `Dockerfile` — builds to `lessandro-nirvana:latest`
- `coplayer_manifest.json` — coworld submission manifest
