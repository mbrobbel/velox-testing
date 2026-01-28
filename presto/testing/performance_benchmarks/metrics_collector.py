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

"""
Metrics collector for Presto queries.

Collects detailed metrics from Presto REST API endpoints after each query
and stores them as parquet files using duckdb.
"""

import json
import subprocess
import requests
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlparse


def collect_metrics(query_id: str, hostname: str, port: int, output_dir: str) -> None:
    """
    Collect metrics from Presto REST API endpoints for a given query.

    Args:
        query_id: The Presto query ID
        hostname: Presto coordinator hostname
        port: Presto coordinator port
        output_dir: Base directory to store metrics
    """
    base_url = f"http://{hostname}:{port}"
    output_path = Path(output_dir) / "metrics" / query_id
    output_path.mkdir(parents=True, exist_ok=True)

    # Collect node information
    _fetch_and_save(f"{base_url}/v1/node", output_path / "nodes")

    # Collect query details and extract stage/task information
    query_info = _fetch_json(f"{base_url}/v1/query/{query_id}")
    if query_info:
        _save_as_json_and_parquet(query_info, output_path / "query")
        _collect_stages(query_info, output_path)
        _collect_worker_data(query_info, output_path)


def _fetch_json(url: str, timeout: int = 30) -> dict | None:
    """Fetch JSON from a URL, returning None on error."""
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Warning: Failed to fetch {url}: {e}")
        return None


def _fetch_and_save(url: str, output_path: Path) -> dict | None:
    """Fetch JSON from URL and save as JSON and parquet."""
    data = _fetch_json(url)
    if data:
        _save_as_json_and_parquet(data, output_path)
    return data


def _save_as_json_and_parquet(data, output_path: Path) -> None:
    """Save data as JSON and convert to parquet."""
    json_file = output_path.with_suffix(".json")
    with open(json_file, "w") as f:
        json.dump(data, f, indent=2)
    _json_to_parquet(json_file, output_path.with_suffix(".parquet"))


def _collect_stages(query_info: dict, output_path: Path) -> None:
    """Extract and save stage information from query info."""
    stages = []
    _extract_stages(query_info.get("outputStage"), stages)

    if stages:
        _save_as_json_and_parquet(stages, output_path / "stages")


def _extract_stages(stage_info: dict, stages: list) -> None:
    """Recursively extract stage data."""
    if stage_info is None:
        return

    latest_attempt = stage_info.get("latestAttemptExecutionInfo", {})
    stages.append({
        "stageId": stage_info.get("stageId"),
        "state": latest_attempt.get("state"),
        "stats": latest_attempt.get("stats"),
    })

    for sub_stage in stage_info.get("subStages", []):
        _extract_stages(sub_stage, stages)


def _collect_worker_data(query_info: dict, output_path: Path) -> None:
    """Collect task details and metrics from each worker."""
    # Group tasks by worker
    tasks_by_worker = defaultdict(list)
    _group_tasks_by_worker(query_info.get("outputStage"), tasks_by_worker)

    all_worker_tasks = {}
    all_worker_metrics = {}

    for worker_uri, task_ids in tasks_by_worker.items():
        worker_id = _worker_id_from_uri(worker_uri)

        # Collect detailed task info from worker
        worker_tasks = []
        for task_id in task_ids:
            task_data = _fetch_json(f"{worker_uri}/v1/task/{task_id}")
            if task_data:
                task_data["_worker_uri"] = worker_uri
                worker_tasks.append(task_data)

        if worker_tasks:
            all_worker_tasks[worker_id] = worker_tasks

        # Collect worker metrics
        metrics = _fetch_worker_metrics(worker_uri)
        if metrics:
            all_worker_metrics[worker_id] = metrics

    if all_worker_tasks:
        _save_as_json_and_parquet(all_worker_tasks, output_path / "tasks")
    if all_worker_metrics:
        _save_as_json_and_parquet(all_worker_metrics, output_path / "metrics")


def _group_tasks_by_worker(stage_info: dict, tasks_by_worker: dict) -> None:
    """Recursively group task IDs by worker URI."""
    if stage_info is None:
        return

    latest_attempt = stage_info.get("latestAttemptExecutionInfo", {})
    for task in latest_attempt.get("tasks", []):
        task_id = task.get("taskId")
        task_self = task.get("taskStatus", {}).get("self")
        if task_id and task_self:
            parsed = urlparse(task_self)
            worker_uri = f"{parsed.scheme}://{parsed.netloc}"
            tasks_by_worker[worker_uri].append(task_id)

    for sub_stage in stage_info.get("subStages", []):
        _group_tasks_by_worker(sub_stage, tasks_by_worker)


def _fetch_worker_metrics(worker_uri: str) -> list | None:
    """Fetch metrics from a worker node."""
    try:
        response = requests.get(f"{worker_uri}/v1/info/metrics", timeout=30)
        response.raise_for_status()

        # Try JSON first, fall back to Prometheus text format
        try:
            data = response.json()
            if not isinstance(data, (dict, list)):
                raise ValueError()
            return data
        except (json.JSONDecodeError, ValueError):
            return _parse_prometheus_metrics(response.text)

    except Exception as e:
        print(f"Warning: Failed to collect metrics from worker {worker_uri}: {e}")
        return None


def _parse_prometheus_metrics(text: str) -> list:
    """Parse Prometheus text format into a list of metric objects."""
    metrics = []
    for line in text.strip().split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.rsplit(" ", 1)
        if len(parts) == 2:
            try:
                value = float(parts[1])
            except ValueError:
                value = parts[1]
            metrics.append({"metric": parts[0], "value": value})
    return metrics or [{"raw": text}]


def _worker_id_from_uri(uri: str) -> str:
    """Extract a filesystem-safe worker ID from a URI."""
    return urlparse(uri).netloc.replace(":", "_").replace(".", "_")


def _json_to_parquet(json_file: Path, parquet_file: Path) -> None:
    """Convert JSON file to parquet using duckdb."""
    try:
        result = subprocess.run(
            ["pixi", "exec", "--spec", "duckdb-cli", "duckdb", "-c",
             f"COPY (SELECT * FROM read_json_auto('{json_file}')) TO '{parquet_file}' (FORMAT PARQUET);"],
            capture_output=True,
            text=True,
            timeout=60
        )
        if result.returncode != 0:
            print(f"Warning: duckdb conversion failed for {json_file}: {result.stderr}")
    except subprocess.TimeoutExpired:
        print(f"Warning: duckdb conversion timed out for {json_file}")
    except Exception as e:
        print(f"Warning: Failed to convert {json_file} to parquet: {e}")
