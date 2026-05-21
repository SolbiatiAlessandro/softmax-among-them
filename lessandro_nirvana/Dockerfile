# lessandro-nirvana Among Them policy
# Buddhist theme: nirvana = liberation, enlightenment, transcendence.
# Strategy: italkalot visual pipeline + Claude via OpenRouter for strategic
# chat and vote reasoning during meetings.
#
# Runtime env vars:
#   COGAMES_ENGINE_WS_URL  - set by the Coworld runner
#   OPENROUTER_API_KEY     - inject via --secret-env at upload time
#   OPENROUTER_MODEL       - optional, default: anthropic/claude-haiku-4-5
#
# Build: This Dockerfile uses a pre-compiled binary approach.
# The Nim source is compiled on the host and the binary is bundled here.
# To recompile from source, see build_from_source.sh.

FROM debian:bookworm-slim

WORKDIR /workspace/bitworld

# Copy game asset files required by the binary (maps, sprites, data)
COPY among_them/*.json ./among_them/
COPY among_them/*.aseprite ./among_them/ 2>/dev/null || true
COPY among_them/*.png ./among_them/ 2>/dev/null || true
COPY clients/data ./clients/data

# Copy the pre-compiled player binary
COPY lessandro_nirvana_binary /bin/lessandro_nirvana
RUN chmod +x /bin/lessandro_nirvana

WORKDIR /workspace/bitworld/among_them/players

# Read COGAMES_ENGINE_WS_URL from the Coworld runner environment.
# OPENROUTER_API_KEY must be injected via --secret-env at coworld upload-policy.
CMD ["/bin/lessandro_nirvana"]
