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
and stores them as JSON files.
"""

import json
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
    _fetch_and_save(f"{base_url}/v1/node", output_path / "nodes.json")

    # Collect query details and extract stage/task information
    query_info = _fetch_json(f"{base_url}/v1/query/{query_id}")
    if query_info:
        _save_json(query_info, output_path / "query.json")
        _collect_stages(query_info, output_path)
        worker_tasks = _collect_worker_data(query_info, output_path)
        _generate_summary(query_info, worker_tasks, output_path)


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
    """Fetch JSON from URL and save to file."""
    data = _fetch_json(url)
    if data:
        _save_json(data, output_path)
    return data


def _save_json(data, output_path: Path) -> None:
    """Save data as JSON."""
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)


def _collect_stages(query_info: dict, output_path: Path) -> None:
    """Extract and save stage information from query info."""
    stages = []
    _extract_stages(query_info.get("outputStage"), stages)

    if stages:
        _save_json(stages, output_path / "stages.json")


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


def _collect_worker_data(query_info: dict, output_path: Path) -> dict:
    """Collect task details and metrics from each worker. Returns worker_tasks dict."""
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
                task_data["_worker_id"] = worker_id
                worker_tasks.append(task_data)

        if worker_tasks:
            all_worker_tasks[worker_id] = worker_tasks

        # Collect worker metrics
        metrics = _fetch_worker_metrics(worker_uri)
        if metrics:
            all_worker_metrics[worker_id] = metrics

    if all_worker_tasks:
        _save_json(all_worker_tasks, output_path / "tasks.json")
    if all_worker_metrics:
        _save_json(all_worker_metrics, output_path / "metrics.json")

    return all_worker_tasks


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


def _generate_summary(query_info: dict, worker_tasks: dict, output_path: Path) -> None:
    """Generate a summary JSON with query -> stages -> workers -> tasks -> pipelines -> operators hierarchy."""
    # Build task lookup by task_id
    task_lookup = {}
    for worker_id, tasks in worker_tasks.items():
        for task in tasks:
            task_lookup[task.get("taskId")] = task

    # Extract stages with workers and tasks
    stages = []
    _extract_stage_summary(query_info.get("outputStage"), task_lookup, stages)

    # Query-level timing
    query_stats = query_info.get("queryStats", {})
    summary = {
        "queryId": query_info.get("queryId"),
        "query": query_info.get("query"),
        "state": query_info.get("state"),
        "createTime": query_stats.get("createTime"),
        "endTime": query_stats.get("endTime"),
        "elapsedTime": query_stats.get("elapsedTime"),
        "executionTime": query_stats.get("executionTime"),
        "queuedTime": query_stats.get("queuedTime"),
        "stages": stages,
    }

    # Save summary (JSON only, no parquet needed for this hierarchical view)
    summary_file = output_path / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)


def _extract_stage_summary(stage_info: dict, task_lookup: dict, stages: list) -> None:
    """Recursively extract stage summary with plan, workers, tasks, pipelines, and operators."""
    if stage_info is None:
        return

    stage_id = stage_info.get("stageId")
    latest_attempt = stage_info.get("latestAttemptExecutionInfo", {})
    stage_stats = latest_attempt.get("stats", {})

    # Extract plan DAG for this stage
    plan = stage_info.get("plan", {})
    plan_dag = _extract_plan_dag(plan.get("root"))

    # Group tasks by worker
    tasks_by_worker = defaultdict(list)
    for task in latest_attempt.get("tasks", []):
        task_id = task.get("taskId")
        task_detail = task_lookup.get(task_id, {})
        worker_id = task_detail.get("_worker_id", "unknown")
        worker_uri = task_detail.get("_worker_uri", "unknown")

        # Task-level timing
        task_stats = task_detail.get("stats", {})

        # Extract pipelines and operators from task stats
        pipelines = []
        for pipeline in task_stats.get("pipelines", []):
            operators = []
            for op in pipeline.get("operatorSummaries", []):
                operators.append({
                    "operatorId": op.get("operatorId"),
                    "operatorType": op.get("operatorType"),
                    "planNodeId": op.get("planNodeId"),
                    "addInputWall": op.get("addInputWall"),
                    "getOutputWall": op.get("getOutputWall"),
                    "finishWall": op.get("finishWall"),
                    "blockedWall": op.get("blockedWall"),
                })
            pipelines.append({
                "pipelineId": pipeline.get("pipelineId"),
                "firstStartTimeMs": pipeline.get("firstStartTimeInMillis"),
                "lastEndTimeMs": pipeline.get("lastEndTimeInMillis"),
                "totalCpuTimeNanos": pipeline.get("totalCpuTimeInNanos"),
                "totalScheduledTimeNanos": pipeline.get("totalScheduledTimeInNanos"),
                "totalBlockedTimeNanos": pipeline.get("totalBlockedTimeInNanos"),
                "operators": operators,
            })

        tasks_by_worker[(worker_id, worker_uri)].append({
            "taskId": task_id,
            "state": task.get("taskStatus", {}).get("state"),
            "createTimeMs": task_stats.get("createTimeInMillis"),
            "firstStartTimeMs": task_stats.get("firstStartTimeInMillis"),
            "endTimeMs": task_stats.get("endTimeInMillis"),
            "elapsedTimeNanos": task_stats.get("elapsedTimeInNanos"),
            "queuedTimeNanos": task_stats.get("queuedTimeInNanos"),
            "totalCpuTimeNanos": task_stats.get("totalCpuTimeInNanos"),
            "totalScheduledTimeNanos": task_stats.get("totalScheduledTimeInNanos"),
            "pipelines": pipelines,
        })

    # Build workers list and collect task times for stage-level aggregation
    workers = []
    all_task_create_times = []
    all_task_end_times = []
    for (worker_id, worker_uri), tasks in tasks_by_worker.items():
        for t in tasks:
            if t.get("createTimeMs"):
                all_task_create_times.append(t["createTimeMs"])
            if t.get("endTimeMs"):
                all_task_end_times.append(t["endTimeMs"])
        workers.append({
            "workerId": worker_id,
            "workerUri": worker_uri,
            "tasks": tasks,
        })

    stages.append({
        "stageId": stage_id,
        "state": latest_attempt.get("state"),
        "startTimeMs": min(all_task_create_times) if all_task_create_times else None,
        "endTimeMs": max(all_task_end_times) if all_task_end_times else None,
        "totalScheduledTime": stage_stats.get("totalScheduledTime"),
        "totalCpuTime": stage_stats.get("totalCpuTime"),
        "totalBlockedTime": stage_stats.get("totalBlockedTime"),
        "plan": plan_dag,
        "workers": workers,
    })

    # Recurse into sub-stages
    for sub_stage in stage_info.get("subStages", []):
        _extract_stage_summary(sub_stage, task_lookup, stages)


def _extract_plan_dag(node: dict) -> dict | None:
    """Recursively extract plan DAG with node types, IDs, and children."""
    if node is None:
        return None

    node_type = node.get("@type", "")
    # Simplify type name (remove package prefix)
    if "." in node_type:
        node_type = node_type.split(".")[-1]

    result = {
        "id": node.get("id"),
        "type": node_type,
    }

    # For RemoteSourceNode, include sourceFragmentIds to link to substages
    if "RemoteSourceNode" in node.get("@type", ""):
        result["sourceFragmentIds"] = node.get("sourceFragmentIds", [])

    # Collect children from various possible fields
    children = []
    if node.get("source"):
        child = _extract_plan_dag(node["source"])
        if child:
            children.append(child)
    for source in node.get("sources", []):
        child = _extract_plan_dag(source)
        if child:
            children.append(child)
    if node.get("left"):
        child = _extract_plan_dag(node["left"])
        if child:
            children.append(child)
    if node.get("right"):
        child = _extract_plan_dag(node["right"])
        if child:
            children.append(child)

    if children:
        result["children"] = children

    return result


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
