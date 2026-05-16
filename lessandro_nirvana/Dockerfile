FROM python:3.11-slim

WORKDIR /app

# Install dependencies
# cogames brings mettagrid (BitWorld protocol) + numpy + websocket support
RUN pip install --no-cache-dir cogames websockets

# Copy policy and player
COPY lessandro_nirvana_policy.py player.py /app/

ENV PYTHONPATH=/app

CMD ["python", "player.py"]
