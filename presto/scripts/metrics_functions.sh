#!/bin/bash

# Copyright (c) 2025, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -e

function get_worker_container_id() {
  # Try GPU worker first, then CPU worker (returns first worker found)
  local container_id=$(docker ps -q --filter "ancestor=presto-native-worker-gpu:latest" | head -1)
  if [[ -z $container_id ]]; then
    container_id=$(docker ps -q --filter "ancestor=presto-native-worker-cpu:latest" | head -1)
  fi
  if [[ -z $container_id ]]; then
    # Fallback: find any container with "native-worker" in the name
    container_id=$(docker ps -q --filter "name=native-worker" | head -1)
  fi
  if [[ -z $container_id ]]; then
    echo "Error: no presto-native worker container found" >&2
    exit 1
  fi
  echo $container_id
}

function get_all_worker_container_ids() {
  # Get all GPU worker containers
  local container_ids=$(docker ps -q --filter "ancestor=presto-native-worker-gpu:latest")
  if [[ -z $container_ids ]]; then
    # Try CPU workers
    container_ids=$(docker ps -q --filter "ancestor=presto-native-worker-cpu:latest")
  fi
  if [[ -z $container_ids ]]; then
    # Fallback: find all containers with "native-worker" in the name
    container_ids=$(docker ps -q --filter "name=native-worker")
  fi
  if [[ -z $container_ids ]]; then
    echo "Error: no presto-native worker containers found" >&2
    exit 1
  fi
  echo $container_ids
}

function get_coordinator_container_id() {
  local container_id=$(docker ps -q --filter "name=presto-coordinator" | head -1)
  if [[ -z $container_id ]]; then
    echo "Error: no presto-coordinator container found" >&2
    exit 1
  fi
  echo $container_id
}

# Convert Prometheus format to JSON
function prometheus_to_json() {
  awk '
    BEGIN { print "{"; first=1 }
    /^[^#]/ && NF >= 2 {
      # Parse metric line: metric_name{labels} value
      line = $0
      if (match(line, /^([a-zA-Z_][a-zA-Z0-9_]*)\{.*\} (.+)$/, arr)) {
        name = arr[1]
        value = arr[2]
      } else if (match(line, /^([a-zA-Z_][a-zA-Z0-9_]*) (.+)$/, arr)) {
        name = arr[1]
        value = arr[2]
      } else {
        next
      }
      if (!first) print ","
      first = 0
      # Check if value is numeric
      if (value ~ /^-?[0-9]+\.?[0-9]*([eE][-+]?[0-9]+)?$/) {
        printf "  \"%s\": %s", name, value
      } else {
        printf "  \"%s\": \"%s\"", name, value
      }
    }
    END { print "\n}" }
  '
}

# Fetch JSON from URL, return empty object on error
function fetch_json() {
  local container_id=$1
  local url=$2
  local result

  result=$(docker exec $container_id curl -s -f "$url" 2>/dev/null) || {
    echo "{}"
    return
  }

  # Check if result looks like JSON (starts with { or [)
  if [[ "$result" =~ ^[[:space:]]*[\{\[] ]]; then
    echo "$result"
  else
    echo "{}"
  fi
}

function collect_metrics() {
  local -r metrics_output_file_path=$1
  local -r coordinator_host=$2
  local -r coordinator_port=$3
  local -r query_id=$4

  local -r worker_container_ids=$(get_all_worker_container_ids)
  local -r coordinator_container_id=$(get_coordinator_container_id)

  # Create a temporary directory for individual metric files
  local -r temp_dir=$(mktemp -d)
  trap "rm -rf $temp_dir" EXIT

  # Collect worker metrics from all workers (Prometheus format) and convert to JSON
  # Also collect worker status to get nodeId and IP mapping
  echo "[" > "$temp_dir/worker_metrics.json"
  local first_worker=true
  for worker_container_id in $worker_container_ids; do
    if [[ "$first_worker" == "true" ]]; then
      first_worker=false
    else
      echo "," >> "$temp_dir/worker_metrics.json"
    fi
    local worker_name=$(docker inspect --format '{{.Name}}' "$worker_container_id" | sed 's/^\///')

    # Get worker status to extract nodeId and IP address
    local worker_status=$(docker exec $worker_container_id curl -s "http://localhost:8080/v1/status" 2>/dev/null || echo "{}")
    local node_id=$(echo "$worker_status" | jq -r '.nodeId // "unknown"')
    local internal_address=$(echo "$worker_status" | jq -r '.internalAddress // "unknown"')

    echo "{\"worker\": \"${worker_name}\", \"nodeId\": \"${node_id}\", \"internalAddress\": \"${internal_address}\", \"metrics\": " >> "$temp_dir/worker_metrics.json"
    docker exec $worker_container_id curl -s \
      "http://localhost:8080/v1/info/metrics" 2>/dev/null | prometheus_to_json >> "$temp_dir/worker_metrics.json" || echo "{}" >> "$temp_dir/worker_metrics.json"
    echo "}" >> "$temp_dir/worker_metrics.json"
  done
  echo "]" >> "$temp_dir/worker_metrics.json"

  # Collect query info from coordinator (includes stages and tasks)
  fetch_json "$coordinator_container_id" "http://localhost:${coordinator_port}/v1/query/${query_id}" > "$temp_dir/query_info.json"

  # Extract task IDs from query info and fetch detailed task info from workers
  local task_ids=$(jq -r '.. | .taskId? // empty' "$temp_dir/query_info.json" 2>/dev/null | sort -u)

  # Collect task info for each task - try all workers since tasks may be distributed
  echo "[" > "$temp_dir/tasks_info.json"
  local first=true
  for task_id in $task_ids; do
    if [[ -n "$task_id" ]]; then
      if [[ "$first" == "true" ]]; then
        first=false
      else
        echo "," >> "$temp_dir/tasks_info.json"
      fi
      # Try each worker until we find the task
      local task_info="{}"
      for worker_container_id in $worker_container_ids; do
        local result=$(fetch_json "$worker_container_id" "http://localhost:8080/v1/task/${task_id}")
        if [[ "$result" != "{}" ]]; then
          task_info="$result"
          break
        fi
      done
      echo "$task_info" >> "$temp_dir/tasks_info.json"
    fi
  done
  echo "]" >> "$temp_dir/tasks_info.json"

  # Combine all metrics into a single JSON file
  jq -n \
    --slurpfile worker_metrics "$temp_dir/worker_metrics.json" \
    --slurpfile query_info "$temp_dir/query_info.json" \
    --slurpfile tasks_info "$temp_dir/tasks_info.json" \
    '{
      "worker_metrics": $worker_metrics[0],
      "query_info": $query_info[0],
      "tasks_info": $tasks_info[0]
    }' > "$metrics_output_file_path"

  if [[ ! -s "$metrics_output_file_path" ]]; then
    echo "Warning: Metrics file is empty" >&2
  fi
}
