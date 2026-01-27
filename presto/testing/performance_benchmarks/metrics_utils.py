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

import subprocess


def collect_metrics(metrics_script_path, metrics_output_file_path, coordinator_host, coordinator_port, query_id):
    """Collect metrics from the presto-native worker and query/task info from coordinator."""
    execute_metrics_function(
        metrics_script_path,
        "collect_metrics",
        metrics_output_file_path,
        coordinator_host,
        coordinator_port,
        query_id
    )


def execute_metrics_function(metrics_script_path, metrics_function, *args):
    args_str = " ".join(f'"{arg}"' for arg in args)
    metrics_command = ["bash", "-c",
                       f"source {metrics_script_path}; {metrics_function} {args_str}"]
    result = subprocess.run(metrics_command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"{metrics_function} returned error code: {result.returncode}, "
            f"stdout: {result.stdout}, stderr: {result.stderr}")
