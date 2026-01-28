#!/usr/bin/env python3
"""
Combines Presto metrics JSON with nsys parquet profile data into a unified JSON file.

Links:
- Plan Node ID: metrics operatorSummaries.planNodeId <-> NVTX_EVENTS.text "[planNodeId]"
- Timestamps: converted using nsys session start time to absolute UTC nanoseconds
- CUDA kernels: correlated with NVTX operator events by overlapping time ranges
"""

import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def run_duckdb_query(query: str) -> str:
    """Run a DuckDB query and return the result as JSON."""
    result = subprocess.run(
        ["pixi", "exec", "--spec", "duckdb-cli", "duckdb", "-json", "-c", query],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"DuckDB query failed: {result.stderr}")
    return result.stdout


def parse_plan_node_id(text: str) -> str | None:
    """Extract plan node ID from NVTX event text like 'CudfFilterProject::getOutput [1756.0]'"""
    match = re.search(r"\[([^\]]+)\]$", text)
    if match:
        return match.group(1)
    return None


def parse_operator_name(text: str) -> str | None:
    """Extract operator name from NVTX event text like 'CudfFilterProject::getOutput [1756.0]'"""
    match = re.match(r"^([^:]+)::(\w+)", text)
    if match:
        return f"{match.group(1)}::{match.group(2)}"
    return None


def main():
    # Paths
    metrics_path = Path(
        "/home/matthijs/velox-testing/presto/scripts/benchmark_output/metrics/tpch/Q9_20260128_113307_00002_5hcyq.json"
    )
    parquet_dir = Path(
        "/home/matthijs/velox-testing/presto/scripts/benchmark_output/q9.parquet"
    )
    output_path = Path(
        "/home/matthijs/velox-testing/presto/scripts/benchmark_output/q9_unified.json"
    )

    # Load metrics JSON
    print("Loading metrics JSON...")
    with open(metrics_path) as f:
        metrics = json.load(f)

    query_info = metrics["query_info"]
    query_stats = query_info["queryStats"]

    # Get session start time from nsys
    print("Getting nsys session info...")
    session_info = json.loads(
        run_duckdb_query(
            f"SELECT * FROM '{parquet_dir}/TARGET_INFO_SESSION_START_TIME.parquet'"
        )
    )[0]
    utc_epoch_ns = int(session_info["utcEpochNs"])
    system_clock_ns = int(session_info["systemClockNs"])

    # Get NVTX operator events (Cudf*)
    print("Querying NVTX events...")
    nvtx_query = f"""
    SELECT
        start,
        "end" as end_time,
        text,
        globalTid,
        domainId
    FROM '{parquet_dir}/NVTX_EVENTS.parquet'
    WHERE text LIKE 'Cudf%' AND "end" IS NOT NULL
    ORDER BY start
    """
    nvtx_events = json.loads(run_duckdb_query(nvtx_query))

    # Get CUDA kernels
    print("Querying CUDA kernels...")
    kernel_query = f"""
    SELECT
        k.start,
        k."end" as end_time,
        k.deviceId,
        k.streamId,
        k.correlationId,
        k.gridX, k.gridY, k.gridZ,
        k.blockX, k.blockY, k.blockZ,
        k.staticSharedMemory,
        k.dynamicSharedMemory,
        k.registersPerThread,
        s.value as kernelName
    FROM '{parquet_dir}/CUPTI_ACTIVITY_KIND_KERNEL.parquet' k
    LEFT JOIN '{parquet_dir}/StringIds.parquet' s ON s.id = k.demangledName
    ORDER BY k.start
    """
    cuda_kernels = json.loads(run_duckdb_query(kernel_query))

    # Get CUDA memcpy
    print("Querying CUDA memcpy...")
    memcpy_query = f"""
    SELECT
        m.start,
        m."end" as end_time,
        m.bytes,
        m.srcKind,
        m.dstKind,
        m.deviceId,
        m.streamId,
        m.correlationId
    FROM '{parquet_dir}/CUPTI_ACTIVITY_KIND_MEMCPY.parquet' m
    ORDER BY m.start
    """
    cuda_memcpy = json.loads(run_duckdb_query(memcpy_query))

    # Get CUDA runtime API calls
    print("Querying CUDA runtime API...")
    runtime_query = f"""
    SELECT
        r.start,
        r."end" as end_time,
        r.globalTid,
        r.correlationId,
        r.returnValue,
        r.callchainId,
        s.value as apiName
    FROM '{parquet_dir}/CUPTI_ACTIVITY_KIND_RUNTIME.parquet' r
    LEFT JOIN '{parquet_dir}/StringIds.parquet' s ON s.id = r.nameId
    ORDER BY r.start
    """
    cuda_runtime = json.loads(run_duckdb_query(runtime_query))

    # Get CUDA GPU memory usage events (alloc/dealloc)
    print("Querying CUDA memory usage events...")
    mem_usage_query = f"""
    SELECT
        m.start,
        m.deviceId,
        m.address,
        m.bytes,
        m.memKind,
        m.memoryOperationType,
        m.correlationId,
        m.streamId,
        m.localMemoryPoolSize,
        m.localMemoryPoolUtilizedSize
    FROM '{parquet_dir}/CUDA_GPU_MEMORY_USAGE_EVENTS.parquet' m
    ORDER BY m.start
    """
    cuda_mem_usage = json.loads(run_duckdb_query(mem_usage_query))

    # Build operator summaries indexed by plan node ID
    print("Building operator index...")
    operator_index = {}
    for op_summary in query_stats["operatorSummaries"]:
        pnid = op_summary["planNodeId"]
        if pnid not in operator_index:
            operator_index[pnid] = []
        operator_index[pnid].append(op_summary)

    # Convert timestamps and enrich NVTX events
    print("Processing NVTX events...")

    def convert_timestamp(nsys_ts: int) -> int:
        """Convert nsys relative timestamp to absolute UTC nanoseconds."""
        return nsys_ts + utc_epoch_ns - system_clock_ns

    enriched_nvtx = []
    for event in nvtx_events:
        plan_node_id = parse_plan_node_id(event["text"])
        operator_name = parse_operator_name(event["text"])

        abs_start = convert_timestamp(int(event["start"]))
        abs_end = convert_timestamp(int(event["end_time"]))

        enriched = {
            "start_ns": abs_start,
            "end_ns": abs_end,
            "duration_ns": abs_end - abs_start,
            "text": event["text"],
            "operator_name": operator_name,
            "plan_node_id": plan_node_id,
            "global_tid": event["globalTid"],
            "domain_id": event["domainId"],
        }

        # Link to metrics operator summary
        if plan_node_id and plan_node_id in operator_index:
            enriched["metrics_summary"] = operator_index[plan_node_id]

        enriched_nvtx.append(enriched)

    # Convert CUDA kernel timestamps and use original nsys timestamps for correlation
    print("Processing CUDA kernels...")
    enriched_kernels = []
    kernel_nsys_times = []  # Keep nsys timestamps for correlation
    for kernel in cuda_kernels:
        nsys_start = int(kernel["start"])
        nsys_end = int(kernel["end_time"])
        abs_start = convert_timestamp(nsys_start)
        abs_end = convert_timestamp(nsys_end)
        enriched_kernels.append(
            {
                "start_ns": abs_start,
                "end_ns": abs_end,
                "duration_ns": abs_end - abs_start,
                "kernel_name": kernel["kernelName"],
                "device_id": kernel["deviceId"],
                "stream_id": kernel["streamId"],
                "correlation_id": kernel["correlationId"],
                "grid": [kernel["gridX"], kernel["gridY"], kernel["gridZ"]],
                "block": [kernel["blockX"], kernel["blockY"], kernel["blockZ"]],
                "shared_memory": {
                    "static": kernel["staticSharedMemory"],
                    "dynamic": kernel["dynamicSharedMemory"],
                },
                "registers_per_thread": kernel["registersPerThread"],
            }
        )
        kernel_nsys_times.append((nsys_start, nsys_end))

    # Convert CUDA memcpy timestamps
    print("Processing CUDA memcpy...")
    enriched_memcpy = []
    memcpy_nsys_times = []  # Keep nsys timestamps for correlation
    for mcpy in cuda_memcpy:
        nsys_start = int(mcpy["start"])
        nsys_end = int(mcpy["end_time"])
        abs_start = convert_timestamp(nsys_start)
        abs_end = convert_timestamp(nsys_end)
        enriched_memcpy.append(
            {
                "start_ns": abs_start,
                "end_ns": abs_end,
                "duration_ns": abs_end - abs_start,
                "bytes": mcpy["bytes"],
                "src_kind": mcpy["srcKind"],
                "dst_kind": mcpy["dstKind"],
                "device_id": mcpy["deviceId"],
                "stream_id": mcpy["streamId"],
                "correlation_id": mcpy["correlationId"],
            }
        )
        memcpy_nsys_times.append((nsys_start, nsys_end))

    # Process CUDA runtime API calls
    print("Processing CUDA runtime API...")
    enriched_runtime = []
    runtime_nsys_times = []
    for api in cuda_runtime:
        nsys_start = int(api["start"])
        nsys_end = int(api["end_time"])
        abs_start = convert_timestamp(nsys_start)
        abs_end = convert_timestamp(nsys_end)
        enriched_runtime.append(
            {
                "start_ns": abs_start,
                "end_ns": abs_end,
                "duration_ns": abs_end - abs_start,
                "api_name": api["apiName"],
                "global_tid": api["globalTid"],
                "correlation_id": api["correlationId"],
                "return_value": api["returnValue"],
                "callchain_id": api["callchainId"],
            }
        )
        runtime_nsys_times.append((nsys_start, nsys_end))

    # Process CUDA memory usage events
    print("Processing CUDA memory usage events...")
    enriched_mem_usage = []
    mem_usage_nsys_times = []
    # Memory operation types: 0 = allocation, 1 = deallocation
    mem_op_names = {0: "allocation", 1: "deallocation"}
    for mem in cuda_mem_usage:
        nsys_start = int(mem["start"])
        abs_start = convert_timestamp(nsys_start)
        enriched_mem_usage.append(
            {
                "start_ns": abs_start,
                "device_id": mem["deviceId"],
                "address": mem["address"],
                "bytes": mem["bytes"],
                "mem_kind": mem["memKind"],
                "operation": mem_op_names.get(mem["memoryOperationType"], str(mem["memoryOperationType"])),
                "correlation_id": mem["correlationId"],
                "stream_id": mem["streamId"],
                "pool_size": mem["localMemoryPoolSize"],
                "pool_utilized": mem["localMemoryPoolUtilizedSize"],
            }
        )
        mem_usage_nsys_times.append(nsys_start)

    # Correlate CUDA activities with NVTX operator events
    print("Correlating CUDA activities with operators...")
    # Build interval tree-like structure for fast lookup (simple approach for now)
    nvtx_intervals = []
    for i, event in enumerate(nvtx_events):
        nvtx_intervals.append(
            (int(event["start"]), int(event["end_time"]), i)
        )
    nvtx_intervals.sort()

    def find_enclosing_nvtx(start_ns: int) -> int | None:
        """Find the index of the NVTX event that encloses the given timestamp."""
        # Simple linear search - could be optimized with interval tree
        for nvtx_start, nvtx_end, idx in nvtx_intervals:
            if nvtx_start <= start_ns <= nvtx_end:
                return idx
        return None

    # Build operator-centric view with correlated CUDA activities
    print("Building operator-centric view...")
    operator_view = {}
    for i, event in enumerate(enriched_nvtx):
        plan_node_id = event["plan_node_id"]
        operator_name = event["operator_name"]
        if plan_node_id not in operator_view:
            operator_view[plan_node_id] = {
                "plan_node_id": plan_node_id,
                "operator_type": operator_name.split("::")[0] if operator_name else None,
                "invocations": [],
                "total_cpu_time_ns": 0,
                "total_kernel_time_ns": 0,
                "total_memcpy_bytes": 0,
                "kernel_count": 0,
                "memcpy_count": 0,
                "runtime_api_count": 0,
                "total_runtime_api_time_ns": 0,
                "mem_alloc_count": 0,
                "mem_dealloc_count": 0,
                "total_mem_allocated_bytes": 0,
                "total_mem_deallocated_bytes": 0,
                "peak_pool_utilized": 0,
            }

        # Find kernels that started during this NVTX event (using original nsys timestamps)
        nvtx_start = int(nvtx_events[i]["start"])
        nvtx_end = int(nvtx_events[i]["end_time"])

        invocation_kernels = []
        for j, (k_start, k_end) in enumerate(kernel_nsys_times):
            if nvtx_start <= k_start <= nvtx_end:
                invocation_kernels.append(j)

        invocation_memcpy = []
        for j, (m_start, m_end) in enumerate(memcpy_nsys_times):
            if nvtx_start <= m_start <= nvtx_end:
                invocation_memcpy.append(j)

        invocation_runtime = []
        for j, (r_start, r_end) in enumerate(runtime_nsys_times):
            if nvtx_start <= r_start <= nvtx_end:
                invocation_runtime.append(j)

        invocation_mem_usage = []
        for j, m_start in enumerate(mem_usage_nsys_times):
            if nvtx_start <= m_start <= nvtx_end:
                invocation_mem_usage.append(j)

        # Calculate memory stats for this invocation
        alloc_bytes = sum(
            enriched_mem_usage[j]["bytes"]
            for j in invocation_mem_usage
            if enriched_mem_usage[j]["operation"] == "allocation"
        )
        dealloc_bytes = sum(
            enriched_mem_usage[j]["bytes"]
            for j in invocation_mem_usage
            if enriched_mem_usage[j]["operation"] == "deallocation"
        )
        alloc_count = sum(
            1 for j in invocation_mem_usage
            if enriched_mem_usage[j]["operation"] == "allocation"
        )
        dealloc_count = sum(
            1 for j in invocation_mem_usage
            if enriched_mem_usage[j]["operation"] == "deallocation"
        )
        peak_pool = max(
            (enriched_mem_usage[j]["pool_utilized"] or 0 for j in invocation_mem_usage),
            default=0
        )

        invocation = {
            "start_ns": event["start_ns"],
            "end_ns": event["end_ns"],
            "duration_ns": event["duration_ns"],
            "method": event["text"].split("::")[1].split()[0] if "::" in event["text"] else None,
            "kernel_indices": invocation_kernels,
            "memcpy_indices": invocation_memcpy,
            "runtime_api_indices": invocation_runtime,
            "mem_usage_indices": invocation_mem_usage,
            "kernel_time_ns": sum(enriched_kernels[j]["duration_ns"] for j in invocation_kernels),
            "memcpy_bytes": sum(enriched_memcpy[j]["bytes"] for j in invocation_memcpy),
            "runtime_api_time_ns": sum(enriched_runtime[j]["duration_ns"] for j in invocation_runtime),
            "mem_allocated_bytes": alloc_bytes,
            "mem_deallocated_bytes": dealloc_bytes,
        }
        operator_view[plan_node_id]["invocations"].append(invocation)
        operator_view[plan_node_id]["total_cpu_time_ns"] += event["duration_ns"]
        operator_view[plan_node_id]["total_kernel_time_ns"] += invocation["kernel_time_ns"]
        operator_view[plan_node_id]["total_memcpy_bytes"] += invocation["memcpy_bytes"]
        operator_view[plan_node_id]["kernel_count"] += len(invocation_kernels)
        operator_view[plan_node_id]["memcpy_count"] += len(invocation_memcpy)
        operator_view[plan_node_id]["runtime_api_count"] += len(invocation_runtime)
        operator_view[plan_node_id]["total_runtime_api_time_ns"] += invocation["runtime_api_time_ns"]
        operator_view[plan_node_id]["mem_alloc_count"] += alloc_count
        operator_view[plan_node_id]["mem_dealloc_count"] += dealloc_count
        operator_view[plan_node_id]["total_mem_allocated_bytes"] += alloc_bytes
        operator_view[plan_node_id]["total_mem_deallocated_bytes"] += dealloc_bytes
        operator_view[plan_node_id]["peak_pool_utilized"] = max(
            operator_view[plan_node_id]["peak_pool_utilized"] or 0, peak_pool or 0
        )

    # Add metrics summary to operator view
    for plan_node_id, op_data in operator_view.items():
        if plan_node_id in operator_index:
            op_data["metrics_summary"] = operator_index[plan_node_id]

    # Parse ISO timestamps from metrics
    create_time = datetime.fromisoformat(
        query_stats["createTime"].replace("Z", "+00:00")
    )
    exec_start = datetime.fromisoformat(
        query_stats["executionStartTime"].replace("Z", "+00:00")
    )
    end_time = datetime.fromisoformat(query_stats["endTime"].replace("Z", "+00:00"))

    # Build unified output
    print("Building unified JSON...")
    unified = {
        "metadata": {
            "query_id": query_info["queryId"],
            "metrics_file": str(metrics_path),
            "parquet_dir": str(parquet_dir),
            "nsys_session_start": {
                "utc_epoch_ns": utc_epoch_ns,
                "utc_time": session_info["utcTime"],
                "local_time": session_info["localTime"],
                "system_clock_ns": system_clock_ns,
            },
        },
        "query_timing": {
            "create_time": query_stats["createTime"],
            "create_time_ns": int(create_time.timestamp() * 1e9),
            "execution_start_time": query_stats["executionStartTime"],
            "execution_start_ns": int(exec_start.timestamp() * 1e9),
            "end_time": query_stats["endTime"],
            "end_time_ns": int(end_time.timestamp() * 1e9),
            "elapsed_time": query_stats["elapsedTime"],
            "queued_time": query_stats["queuedTime"],
            "analysis_time": query_stats["analysisTime"],
            "total_planning_time": query_stats["totalPlanningTime"],
            "finishing_time": query_stats["finishingTime"],
            "total_cpu_time": query_stats["totalCpuTime"],
            "total_scheduled_time": query_stats["totalScheduledTime"],
            "total_blocked_time": query_stats["totalBlockedTime"],
        },
        "query_resources": {
            "total_tasks": query_stats["totalTasks"],
            "completed_tasks": query_stats["completedTasks"],
            "total_drivers": query_stats["totalDrivers"],
            "completed_drivers": query_stats["completedDrivers"],
            "total_splits": query_stats["totalSplits"],
            "completed_splits": query_stats["completedSplits"],
            "peak_user_memory": query_stats["peakUserMemoryReservation"],
            "peak_total_memory": query_stats["peakTotalMemoryReservation"],
            "raw_input_data_size": query_stats["rawInputDataSize"],
            "raw_input_positions": query_stats["rawInputPositions"],
            "output_data_size": query_stats["outputDataSize"],
            "output_positions": query_stats["outputPositions"],
        },
        "operator_summaries": query_stats["operatorSummaries"],
        "plan": json.loads(query_info["outputStage"]["plan"]["jsonRepresentation"]),
        "nsys_profile": {
            "nvtx_operator_events": enriched_nvtx,
            "cuda_kernels": enriched_kernels,
            "cuda_memcpy": enriched_memcpy,
            "cuda_runtime_api": enriched_runtime,
            "cuda_mem_usage": enriched_mem_usage,
            "summary": {
                "nvtx_events_count": len(enriched_nvtx),
                "cuda_kernels_count": len(enriched_kernels),
                "cuda_memcpy_count": len(enriched_memcpy),
                "cuda_runtime_api_count": len(enriched_runtime),
                "cuda_mem_usage_count": len(enriched_mem_usage),
                "total_kernel_time_ns": sum(k["duration_ns"] for k in enriched_kernels),
                "total_memcpy_bytes": sum(m["bytes"] for m in enriched_memcpy),
                "total_runtime_api_time_ns": sum(r["duration_ns"] for r in enriched_runtime),
                "total_mem_allocated_bytes": sum(
                    m["bytes"] for m in enriched_mem_usage if m["operation"] == "allocation"
                ),
                "total_mem_deallocated_bytes": sum(
                    m["bytes"] for m in enriched_mem_usage if m["operation"] == "deallocation"
                ),
            },
        },
        "operator_view": operator_view,
    }

    # Write output
    print(f"Writing unified JSON to {output_path}...")
    with open(output_path, "w") as f:
        json.dump(unified, f, indent=2)

    print(f"Done! Output: {output_path}")
    print(f"  - NVTX operator events: {len(enriched_nvtx)}")
    print(f"  - CUDA kernels: {len(enriched_kernels)}")
    print(f"  - CUDA memcpy: {len(enriched_memcpy)}")
    print(f"  - CUDA runtime API: {len(enriched_runtime)}")
    print(f"  - CUDA memory usage: {len(enriched_mem_usage)}")


if __name__ == "__main__":
    main()
