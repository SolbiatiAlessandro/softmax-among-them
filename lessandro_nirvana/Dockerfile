# lessandro-nirvana: LLM-enhanced Among Them policy using OpenRouter.
# Wraps italkalot.nim with an OpenRouter (Claude) API adapter.
# Set OPENROUTER_API_KEY at runtime for LLM-enhanced voting.
FROM alpine AS build

ARG BITWORLD_REPO=https://github.com/Metta-AI/bitworld.git
ARG BITWORLD_REF=b87c58ce1ba95d7dac12820bc21c3b7d58722869

RUN apk add --no-cache build-base curl gcompat git

RUN if [ "$(apk --print-arch)" = "x86_64" ]; then \
    curl -fsSL \
      -o /usr/local/bin/nimby \
https://github.com/treeform/nimby/releases/download/0.1.26/nimby-Linux-X64; \
  elif [ "$(apk --print-arch)" = "aarch64" ]; then \
    curl -fsSL \
      -o /usr/local/bin/nimby \
https://github.com/treeform/nimby/releases/download/0.1.26/nimby-Linux-ARM64; \
  else \
    echo "unsupported arch: $(apk --print-arch)" && exit 1; \
  fi && \
  chmod +x /usr/local/bin/nimby && \
  nimby use 2.2.4

ENV PATH="/root/.nimby/nim/bin:$PATH"

WORKDIR /workspace
RUN git clone "$BITWORLD_REPO" bitworld
WORKDIR /workspace/bitworld
RUN git checkout "$BITWORLD_REF"

RUN nimby --global sync nimby.lock && \
  cat nim.cfg >> /root/.nimby/nim/config/nim.cfg

# Replace OpenAI client with OpenRouter-compatible adapter
COPY openrouter.nim src/bitworld/ais/openai.nim

# Copy our main policy into the italkalot player directory
COPY lessandro_nirvana.nim among_them/players/italkalot/italkalot.nim
WORKDIR /workspace/bitworld/among_them/players/italkalot
RUN nim c \
  -d:release \
  -d:botHeadless \
  --nimcache:/tmp/bitworld-nimcache \
  --out:lessandro_nirvana \
  italkalot.nim

FROM alpine

WORKDIR /workspace/bitworld
COPY --from=build \
  /workspace/bitworld/among_them/players/italkalot/lessandro_nirvana \
  /bin/lessandro_nirvana
COPY --from=build /workspace/bitworld/among_them/*.json ./among_them/
COPY --from=build /workspace/bitworld/among_them/*.aseprite ./among_them/
COPY --from=build /workspace/bitworld/among_them/*.png ./among_them/
COPY --from=build /workspace/bitworld/clients/data ./clients/data

WORKDIR /workspace/bitworld/among_them/players
ENV OPENROUTER_API_KEY=""
CMD ["/bin/lessandro_nirvana", "--address:host.docker.internal", "--port:8080"]
