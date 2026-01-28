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

import prestodb
import pytest

from datetime import datetime
from pathlib import Path
from .benchmark_keys import BenchmarkKeys
from .profiler_utils import start_profiler, stop_profiler
from .metrics_utils import collect_metrics
from ..common.fixtures import tpch_queries, tpcds_queries


@pytest.fixture(scope="module")
def presto_cursor(request):
    hostname = request.config.getoption("--hostname")
    port = request.config.getoption("--port")
    user = request.config.getoption("--user")
    schema = request.config.getoption("--schema-name")
    conn = prestodb.dbapi.connect(host=hostname, port=port, user=user, catalog="hive",
                                  schema=schema)
    return conn.cursor()


@pytest.fixture(scope="session")
def benchmark_result_collector(request):
    benchmark_results = {}
    yield benchmark_results

    request.session.benchmark_results = benchmark_results


@pytest.fixture(scope="module")
def benchmark_queries(request, tpch_queries, tpcds_queries):
    if request.node.obj.BENCHMARK_TYPE == "tpch":
        return tpch_queries
    else:
        assert request.node.obj.BENCHMARK_TYPE == "tpcds"
        return tpcds_queries


@pytest.fixture(scope="module")
def benchmark_query(request, presto_cursor, benchmark_queries, benchmark_result_collector):
    iterations = request.config.getoption("--iterations")
    profile = request.config.getoption("--profile")
    profile_script_path = request.config.getoption("--profile-script-path")
    metrics = request.config.getoption("--metrics")
    metrics_script_path = request.config.getoption("--metrics-script-path")
    hostname = request.config.getoption("--hostname")
    port = request.config.getoption("--port")
    benchmark_type = request.node.obj.BENCHMARK_TYPE
    bench_output_dir = request.config.getoption("--output-dir")

    if profile:
        assert profile_script_path is not None
        profile_output_dir_path = Path(f"{bench_output_dir}/profiles/{benchmark_type}")
        profile_output_dir_path.mkdir(parents=True, exist_ok=True)

    if metrics:
        assert metrics_script_path is not None
        metrics_output_dir_path = Path(f"{bench_output_dir}/metrics/{benchmark_type}")
        metrics_output_dir_path.mkdir(parents=True, exist_ok=True)

    benchmark_result_collector[benchmark_type] = {
        BenchmarkKeys.RAW_TIMES_KEY: {},
        BenchmarkKeys.FAILED_QUERIES_KEY: {},
    }

    benchmark_dict = benchmark_result_collector[benchmark_type]
    raw_times_dict = benchmark_dict[BenchmarkKeys.RAW_TIMES_KEY]
    assert raw_times_dict == {}

    failed_queries_dict = benchmark_dict[BenchmarkKeys.FAILED_QUERIES_KEY]
    assert failed_queries_dict == {}

    def benchmark_query_function(query_id):
        last_presto_query_id = None
        try:
            if profile:
                profile_output_file_path = f"{profile_output_dir_path.absolute()}/{query_id}.nsys-rep"
                start_profiler(profile_script_path, profile_output_file_path)
            result = []
            for i in range(iterations):
                presto_cursor.execute("--" + str(benchmark_type) + "_" + str(query_id) + "--" + "\n" +
                                      benchmark_queries[query_id])
                result.append(presto_cursor.stats["elapsedTimeMillis"])
                last_presto_query_id = presto_cursor.stats.get("queryId")
            raw_times_dict[query_id] = result
        except Exception as e:
            # Try to get queryId from stats even on failure
            if presto_cursor.stats:
                last_presto_query_id = presto_cursor.stats.get("queryId")
            failed_queries_dict[query_id] = f"{e.error_type}: {e.error_name}"
            raw_times_dict[query_id] = None
            raise
        finally:
            if profile:
                stop_profiler(profile_script_path, profile_output_file_path)
            if metrics and last_presto_query_id:
                # Use Presto query ID in filename for easy identification
                metrics_output_file_path = f"{metrics_output_dir_path.absolute()}/{query_id}_{last_presto_query_id}.json"
                try:
                    collect_metrics(metrics_script_path, metrics_output_file_path,
                                    hostname, port, last_presto_query_id)
                except Exception as e:
                    print(f"Warning: Failed to collect metrics for {query_id}: {e}")

    return benchmark_query_function
