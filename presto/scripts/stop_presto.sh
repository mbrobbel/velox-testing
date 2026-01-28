#!/bin/bash

set -e

# Include multi-worker compose file if it exists
MULTI_WORKER_COMPOSE=""
if [[ -f ../docker/docker-compose.workers.yml ]]; then
  MULTI_WORKER_COMPOSE="-f ../docker/docker-compose.workers.yml"
fi

# Use --profile to ensure single-worker profile containers are also stopped
docker compose -f ../docker/docker-compose.java.yml -f ../docker/docker-compose.native-cpu.yml -f ../docker/docker-compose.native-gpu.yml --profile single-worker ${MULTI_WORKER_COMPOSE} down
