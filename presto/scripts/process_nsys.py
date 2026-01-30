#!/usr/bin/env python3
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
Process nsys parquet export data and generate JSON for timeline visualization.

Extracts:
- Memory allocation/deallocation events aggregated over time
- Host-to-Device and Device-to-Host memcpy throughput over time
"""

import argparse
import json
import subprocess
from pathlib import Path


def run_duckdb_query(parquet_dir: Path, query: str) -> list[dict]:
    """Run a DuckDB query and return results as list of dicts."""
    # Use pixi to run duckdb-cli
    cmd = ["pixi", "exec", "--spec", "duckdb-cli", "duckdb", "-json", "-c", query]
    result = subprocess.run(cmd, cwd=parquet_dir, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"DuckDB error: {result.stderr}")
        return []
    if not result.stdout.strip():
        return []
    return json.loads(result.stdout)


def process_nsys_data(parquet_dir: Path, bucket_size_ms: int = 100) -> dict:
    """
    Process nsys parquet data and return aggregated metrics.

    Args:
        parquet_dir: Path to the nsys parquet export directory
        bucket_size_ms: Time bucket size in milliseconds for aggregation

    Returns:
        Dictionary with nsys metrics for timeline visualization
    """
    bucket_size_ns = bucket_size_ms * 1_000_000

    # Get session start time info
    session_info = run_duckdb_query(parquet_dir,
        "SELECT utcEpochNs, utcTime FROM 'TARGET_INFO_SESSION_START_TIME.parquet'")

    session_start_utc_ns = int(session_info[0]["utcEpochNs"]) if session_info else 0
    session_start_time = session_info[0]["utcTime"] if session_info else "unknown"

    # Get time range from memcpy events
    time_range = run_duckdb_query(parquet_dir,
        "SELECT min(start) as min_start, max(\"end\") as max_end FROM 'CUPTI_ACTIVITY_KIND_MEMCPY.parquet'")

    if not time_range or time_range[0]["min_start"] is None:
        # Fall back to memory events
        time_range = run_duckdb_query(parquet_dir,
            "SELECT min(start) as min_start, max(start) as max_end FROM 'CUDA_GPU_MEMORY_USAGE_EVENTS.parquet'")

    min_ns = int(time_range[0]["min_start"]) if time_range else 0
    max_ns = int(time_range[0]["max_end"]) if time_range else 0

    # Query memcpy data aggregated by time bucket and direction
    # copyKind: 1 = HtoD, 2 = DtoH
    # Use floor division to bucket by time
    memcpy_query = f"""
    SELECT
        floor(start / {bucket_size_ns}.0)::BIGINT * {bucket_size_ms} as bucket_ms,
        copyKind,
        count(*) as count,
        sum(bytes)::BIGINT as total_bytes,
        sum(\"end\" - start)::BIGINT as total_duration_ns
    FROM 'CUPTI_ACTIVITY_KIND_MEMCPY.parquet'
    WHERE copyKind IN (1, 2)
    GROUP BY bucket_ms, copyKind
    ORDER BY bucket_ms, copyKind
    """
    memcpy_data = run_duckdb_query(parquet_dir, memcpy_query)

    # Query memory allocation events aggregated by time bucket
    # memoryOperationType: 0 = allocation, 1 = deallocation
    memory_query = f"""
    SELECT
        floor(start / {bucket_size_ns}.0)::BIGINT * {bucket_size_ms} as bucket_ms,
        memoryOperationType as op_type,
        count(*) as count,
        sum(bytes)::BIGINT as total_bytes
    FROM 'CUDA_GPU_MEMORY_USAGE_EVENTS.parquet'
    GROUP BY bucket_ms, op_type
    ORDER BY bucket_ms, op_type
    """
    memory_data = run_duckdb_query(parquet_dir, memory_query)

    # Query local memory pool size and utilization over time
    pool_query = f"""
    SELECT
        floor(start / {bucket_size_ns}.0)::BIGINT * {bucket_size_ms} as bucket_ms,
        max(localMemoryPoolSize)::BIGINT as pool_size,
        max(localMemoryPoolUtilizedSize)::BIGINT as pool_utilized
    FROM 'CUDA_GPU_MEMORY_USAGE_EVENTS.parquet'
    WHERE localMemoryPoolSize > 0
    GROUP BY bucket_ms
    ORDER BY bucket_ms
    """
    pool_data = run_duckdb_query(parquet_dir, pool_query)

    # Process memcpy into time series
    htod_throughput = []  # Host to Device
    dtoh_throughput = []  # Device to Host

    for row in memcpy_data:
        bucket_ms = int(row["bucket_ms"])
        total_bytes = int(row["total_bytes"])
        # Throughput in GB/s (bytes per bucket_size_ms)
        throughput_gbps = (total_bytes / 1e9) / (bucket_size_ms / 1000)

        entry = {
            "timeMs": bucket_ms,
            "bytes": total_bytes,
            "count": int(row["count"]),
            "throughputGBps": round(throughput_gbps, 3)
        }

        if int(row["copyKind"]) == 1:
            htod_throughput.append(entry)
        elif int(row["copyKind"]) == 2:
            dtoh_throughput.append(entry)

    # Process memory allocations into time series
    allocations = []
    deallocations = []

    for row in memory_data:
        bucket_ms = int(row["bucket_ms"])
        entry = {
            "timeMs": bucket_ms,
            "bytes": int(row["total_bytes"]),
            "count": int(row["count"])
        }

        if int(row["op_type"]) == 0:
            allocations.append(entry)
        else:
            deallocations.append(entry)

    # Calculate cumulative memory usage over time
    # Combine allocations and deallocations, sort by time
    all_mem_events = []
    for row in memory_data:
        bucket_ms = int(row["bucket_ms"])
        bytes_val = int(row["total_bytes"])
        if int(row["op_type"]) == 1:  # deallocation
            bytes_val = -bytes_val
        all_mem_events.append((bucket_ms, bytes_val))

    all_mem_events.sort(key=lambda x: x[0])

    cumulative_memory = []
    running_total = 0
    for time_ms, bytes_val in all_mem_events:
        running_total += bytes_val
        # Find or create entry for this time bucket
        if cumulative_memory and cumulative_memory[-1]["timeMs"] == time_ms:
            cumulative_memory[-1]["bytesAllocated"] = running_total
        else:
            cumulative_memory.append({
                "timeMs": time_ms,
                "bytesAllocated": running_total
            })

    # Process memory pool data
    memory_pool = []
    for row in pool_data:
        memory_pool.append({
            "timeMs": int(row["bucket_ms"]),
            "poolSize": int(row["pool_size"]),
            "poolUtilized": int(row["pool_utilized"]),
        })

    return {
        "sessionStartUtcNs": session_start_utc_ns,
        "sessionStartTime": session_start_time,
        "timeRangeNs": {"min": min_ns, "max": max_ns},
        "timeRangeMs": {"min": min_ns / 1_000_000, "max": max_ns / 1_000_000},
        "bucketSizeMs": bucket_size_ms,
        "htodThroughput": htod_throughput,
        "dtohThroughput": dtoh_throughput,
        "allocations": allocations,
        "deallocations": deallocations,
        "cumulativeMemory": cumulative_memory,
        "memoryPool": memory_pool,
    }


def main():
    parser = argparse.ArgumentParser(description="Process nsys parquet data for timeline visualization")
    parser.add_argument("parquet_dir", type=Path, help="Path to nsys parquet export directory")
    parser.add_argument("-o", "--output", type=Path, help="Output JSON file path")
    parser.add_argument("--bucket-size", type=int, default=100, help="Time bucket size in milliseconds (default: 100)")
    args = parser.parse_args()

    if not args.parquet_dir.exists():
        print(f"Error: {args.parquet_dir} does not exist")
        return 1

    output_path = args.output or args.parquet_dir.parent / "nsys_metrics.json"

    print(f"Processing nsys data from {args.parquet_dir}...")
    metrics = process_nsys_data(args.parquet_dir, args.bucket_size)

    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Wrote nsys metrics to {output_path}")
    print(f"  Time range: {metrics['timeRangeMs']['min']:.2f}ms - {metrics['timeRangeMs']['max']:.2f}ms")
    print(f"  HtoD samples: {len(metrics['htodThroughput'])}")
    print(f"  DtoH samples: {len(metrics['dtohThroughput'])}")
    print(f"  Memory pool samples: {len(metrics['memoryPool'])}")
    print(f"  Memory samples: {len(metrics['cumulativeMemory'])}")

    return 0


if __name__ == "__main__":
    exit(main())
