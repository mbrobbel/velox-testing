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
Generate an interactive HTML timeline visualization from a Presto query summary.json file.
"""

import argparse
import json
from pathlib import Path


HTML_TEMPLATE = '''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Query Timeline: {query_id}</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #1a1a2e; color: #eee; padding: 20px; }}
        h1 {{ margin-bottom: 10px; font-size: 1.4em; color: #fff; }}
        .query-text {{ font-family: monospace; font-size: 0.85em; color: #888; margin-bottom: 20px; max-height: 60px; overflow: auto; white-space: pre-wrap; background: #16213e; padding: 10px; border-radius: 4px; }}
        .stats {{ display: flex; gap: 20px; margin-bottom: 20px; font-size: 0.9em; }}
        .stat {{ background: #16213e; padding: 8px 12px; border-radius: 4px; }}
        .stat-label {{ color: #888; }}
        .stat-value {{ color: #4fc3f7; font-weight: bold; }}
        .timeline-container {{ position: relative; overflow-x: auto; background: #16213e; border-radius: 8px; padding: 15px; }}
        .timeline {{ position: relative; min-height: 100px; }}
        .time-axis {{ height: 30px; position: relative; border-bottom: 1px solid #333; margin-bottom: 10px; margin-left: 360px; margin-right: 10px; }}
        .time-tick {{ position: absolute; top: 0; height: 100%; border-left: 1px solid #333; }}
        .time-label {{ position: absolute; top: 5px; font-size: 0.75em; color: #888; transform: translateX(-50%); }}
        .row {{ position: relative; height: 28px; margin: 2px 0; }}
        .row.stage {{ height: 32px; margin-top: 10px; }}
        .row.worker {{ height: 28px; }}
        .row.task {{ height: 24px; }}
        .row.pipeline {{ height: 20px; }}
        .row.operator {{ height: 18px; }}
        .label {{ position: absolute; left: 0; width: 350px; padding-right: 10px; font-size: 0.8em; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; line-height: 28px; }}
        .row.stage .label {{ font-weight: bold; color: #4fc3f7; line-height: 32px; }}
        .row.worker .label {{ padding-left: 15px; color: #81c784; }}
        .row.task .label {{ padding-left: 30px; color: #fff; font-size: 0.75em; line-height: 24px; }}
        .row.pipeline .label {{ padding-left: 45px; color: #4fc3f7; font-size: 0.75em; line-height: 20px; font-family: monospace; background: transparent; }}
        .pipeline-group {{ border-left: 2px solid #4fc3f7; margin-left: 45px; padding-left: 5px; margin-bottom: 4px; }}
        .row.operator .label {{ padding-left: 45px; color: #aaa; font-size: 0.7em; line-height: 18px; font-family: monospace; }}
        .row.operator .label.op-scan {{ color: #4caf50; }}
        .row.operator .label.op-filter {{ color: #ff9800; }}
        .row.operator .label.op-project {{ color: #ff9800; }}
        .row.operator .label.op-join {{ color: #e91e63; }}
        .row.operator .label.op-agg {{ color: #9c27b0; }}
        .row.operator .label.op-sort {{ color: #673ab7; }}
        .row.operator .label.op-exchange {{ color: #00bcd4; }}
        .row.operator .label.op-output {{ color: #f44336; }}
        .bar-container {{ position: absolute; left: 360px; right: 10px; height: 100%; }}
        .bar {{ position: absolute; height: 70%; top: 15%; border-radius: 3px; min-width: 2px; cursor: pointer; transition: filter 0.15s; }}
        .bar:hover {{ filter: brightness(1.3); }}
        .row.stage .bar {{ background: linear-gradient(90deg, #1565c0, #1976d2); height: 80%; top: 10%; }}
        .row.worker .bar {{ background: linear-gradient(90deg, #2e7d32, #43a047); }}
        .row.task .bar {{ background: linear-gradient(90deg, #6a1b9a, #8e24aa); height: 65%; top: 17%; }}
        .row.pipeline .bar {{ background: linear-gradient(90deg, #e65100, #f57c00); height: 60%; top: 20%; }}
        .row.pipeline .bar.active {{ background: linear-gradient(90deg, #ff8f00, #ffb300); height: 40%; top: 30%; }}
        .row.operator .bar {{ height: 55%; top: 22%; }}
        .row.operator .bar.add-input {{ background: linear-gradient(90deg, #00695c, #00897b); }}
        .row.operator .bar.get-output {{ background: linear-gradient(90deg, #1565c0, #1e88e5); }}
        .row.operator .bar.finish {{ background: linear-gradient(90deg, #6a1b9a, #8e24aa); }}
        .row.operator .bar.blocked {{ background: linear-gradient(90deg, #c62828, #e53935); }}
        .tooltip {{ position: fixed; background: #222; color: #fff; padding: 10px 14px; border-radius: 6px; font-size: 0.85em; pointer-events: none; z-index: 1000; max-width: 400px; box-shadow: 0 4px 12px rgba(0,0,0,0.4); display: none; }}
        .tooltip-title {{ font-weight: bold; margin-bottom: 6px; color: #4fc3f7; }}
        .tooltip-row {{ display: flex; justify-content: space-between; gap: 20px; margin: 2px 0; }}
        .tooltip-label {{ color: #888; }}
        .tooltip-value {{ color: #fff; font-family: monospace; }}
        .toggle {{ cursor: pointer; user-select: none; }}
        .toggle::before {{ content: "▼ "; font-size: 0.7em; }}
        .toggle.collapsed::before {{ content: "▶ "; }}
        .collapsed-children {{ display: none; }}
        .legend {{ display: flex; gap: 20px; margin-bottom: 15px; font-size: 0.85em; }}
        .legend-item {{ display: flex; align-items: center; gap: 6px; }}
        .legend-color {{ width: 16px; height: 16px; border-radius: 3px; }}
        .legend-color.stage {{ background: #1976d2; }}
        .legend-color.worker {{ background: #43a047; }}
        .legend-color.task {{ background: #8e24aa; }}
        .legend-color.pipeline {{ background: #f57c00; }}
        .gpu-charts {{ margin-top: 20px; }}
        .gpu-chart {{ background: #16213e; border-radius: 8px; padding: 15px; margin-bottom: 15px; }}
        .gpu-chart h3 {{ margin: 0 0 10px 0; font-size: 1em; color: #4fc3f7; }}
        .chart-container {{ position: relative; height: 100px; margin-left: 360px; margin-right: 10px; }}
        .chart-canvas {{ width: 100%; height: 100%; }}
        .chart-y-axis {{ position: absolute; left: -110px; top: 0; bottom: 0; width: 100px; display: flex; flex-direction: column; justify-content: space-between; font-size: 0.7em; color: #888; text-align: right; padding-right: 5px; }}
        .chart-area {{ position: relative; height: 100%; background: rgba(0,0,0,0.2); border-radius: 4px; overflow: hidden; }}
        .chart-bar {{ position: absolute; bottom: 0; background: linear-gradient(180deg, rgba(79, 195, 247, 0.8), rgba(79, 195, 247, 0.3)); min-width: 1px; }}
        .chart-bar.htod {{ background: linear-gradient(180deg, rgba(76, 175, 80, 0.8), rgba(76, 175, 80, 0.3)); }}
        .chart-bar.dtoh {{ background: linear-gradient(180deg, rgba(255, 152, 0, 0.8), rgba(255, 152, 0, 0.3)); }}
        .chart-bar.memory {{ background: linear-gradient(180deg, rgba(156, 39, 176, 0.8), rgba(156, 39, 176, 0.3)); }}
        .chart-line {{ position: absolute; bottom: 0; border-top: 2px solid #4fc3f7; }}
        .chart-line.memory {{ border-color: #9c27b0; }}
        .gpu-legend {{ display: flex; gap: 15px; margin-top: 8px; font-size: 0.8em; }}
        .gpu-legend-item {{ display: flex; align-items: center; gap: 5px; }}
        .gpu-legend-color {{ width: 12px; height: 12px; border-radius: 2px; }}
    </style>
</head>
<body>
    <h1>Query Timeline: {query_id}</h1>
    <div class="query-text">{query_text}</div>
    <div class="stats">
        <div class="stat"><span class="stat-label">State:</span> <span class="stat-value">{state}</span></div>
        <div class="stat"><span class="stat-label">Elapsed:</span> <span class="stat-value">{elapsed_time}</span></div>
        <div class="stat"><span class="stat-label">Execution:</span> <span class="stat-value">{execution_time}</span></div>
        <div class="stat"><span class="stat-label">Queued:</span> <span class="stat-value">{queued_time}</span></div>
    </div>
    <div class="legend">
        <div class="legend-item"><div class="legend-color stage"></div>Stage</div>
        <div class="legend-item"><div class="legend-color worker"></div>Worker</div>
        <div class="legend-item"><div class="legend-color task"></div>Task</div>
        <div class="legend-item"><div class="legend-color pipeline"></div>Pipeline</div>
        <div class="legend-item"><div class="legend-color" style="background:#00897b"></div>AddInput</div>
        <div class="legend-item"><div class="legend-color" style="background:#1e88e5"></div>GetOutput</div>
        <div class="legend-item"><div class="legend-color" style="background:#8e24aa"></div>Finish</div>
        <div class="legend-item"><div class="legend-color" style="background:#e53935"></div>Blocked</div>
    </div>
    <div class="timeline-container">
        <div class="time-axis" id="time-axis"></div>
        <div class="timeline" id="timeline"></div>
    </div>
    <div class="gpu-charts" id="gpu-charts"></div>
    <div class="tooltip" id="tooltip"></div>

    <script>
    const data = {data_json};
    const nsysData = {nsys_json};
    const timelineEl = document.getElementById('timeline');
    const timeAxisEl = document.getElementById('time-axis');
    const tooltipEl = document.getElementById('tooltip');

    // Find global time range
    let minTime = Infinity, maxTime = -Infinity;
    data.stages.forEach(stage => {{
        if (stage.startTimeMs) minTime = Math.min(minTime, stage.startTimeMs);
        if (stage.endTimeMs) maxTime = Math.max(maxTime, stage.endTimeMs);
        (stage.workers || []).forEach(worker => {{
            (worker.tasks || []).forEach(task => {{
                if (task.createTimeMs) minTime = Math.min(minTime, task.createTimeMs);
                if (task.endTimeMs) maxTime = Math.max(maxTime, task.endTimeMs);
            }});
        }});
    }});

    const duration = maxTime - minTime || 1;
    const toPercent = (ms) => ((ms - minTime) / duration * 100);

    // Draw time axis
    const numTicks = 10;
    for (let i = 0; i <= numTicks; i++) {{
        const pct = i / numTicks * 100;
        const ms = minTime + (i / numTicks) * duration;
        const tick = document.createElement('div');
        tick.className = 'time-tick';
        tick.style.left = pct + '%';
        timeAxisEl.appendChild(tick);

        const label = document.createElement('div');
        label.className = 'time-label';
        label.style.left = pct + '%';
        label.textContent = (ms - minTime).toFixed(0) + 'ms';
        timeAxisEl.appendChild(label);
    }}

    function formatNanos(ns) {{
        if (ns == null) return '-';
        if (ns < 1000) return ns + 'ns';
        if (ns < 1000000) return (ns / 1000).toFixed(2) + 'µs';
        if (ns < 1000000000) return (ns / 1000000).toFixed(2) + 'ms';
        return (ns / 1000000000).toFixed(2) + 's';
    }}

    function parseTimeString(str) {{
        // Parse Presto time strings like "5.23ms", "1.20s", "500us", "200ns"
        if (!str) return 0;
        const match = str.match(/^([\\d.]+)(ns|us|µs|ms|s|m|h|d)$/);
        if (!match) return 0;
        const val = parseFloat(match[1]);
        const unit = match[2];
        switch (unit) {{
            case 'ns': return val / 1000000;
            case 'us': case 'µs': return val / 1000;
            case 'ms': return val;
            case 's': return val * 1000;
            case 'm': return val * 60000;
            case 'h': return val * 3600000;
            case 'd': return val * 86400000;
            default: return 0;
        }}
    }}

    function showTooltip(e, title, rows) {{
        let html = '<div class="tooltip-title">' + title + '</div>';
        rows.forEach(([label, value]) => {{
            html += '<div class="tooltip-row"><span class="tooltip-label">' + label + '</span><span class="tooltip-value">' + value + '</span></div>';
        }});
        tooltipEl.innerHTML = html;
        tooltipEl.style.display = 'block';
        tooltipEl.style.left = (e.clientX + 15) + 'px';
        tooltipEl.style.top = (e.clientY + 15) + 'px';
    }}

    function hideTooltip() {{
        tooltipEl.style.display = 'none';
    }}

    function createRow(type, label, startMs, endMs, tooltipData, children) {{
        const row = document.createElement('div');
        row.className = 'row ' + type;

        const labelEl = document.createElement('div');
        labelEl.className = 'label';
        if (children && children.length > 0) {{
            labelEl.className += ' toggle';
            labelEl.onclick = () => {{
                labelEl.classList.toggle('collapsed');
                childContainer.classList.toggle('collapsed-children');
            }};
        }}
        labelEl.textContent = label;
        row.appendChild(labelEl);

        const barContainer = document.createElement('div');
        barContainer.className = 'bar-container';

        if (startMs != null && endMs != null) {{
            const bar = document.createElement('div');
            bar.className = 'bar';
            bar.style.left = toPercent(startMs) + '%';
            bar.style.width = Math.max(0.5, toPercent(endMs) - toPercent(startMs)) + '%';
            bar.onmouseenter = (e) => showTooltip(e, tooltipData.title, tooltipData.rows);
            bar.onmousemove = (e) => {{ tooltipEl.style.left = (e.clientX + 15) + 'px'; tooltipEl.style.top = (e.clientY + 15) + 'px'; }};
            bar.onmouseleave = hideTooltip;
            barContainer.appendChild(bar);
        }}

        row.appendChild(barContainer);
        timelineEl.appendChild(row);

        if (children && children.length > 0) {{
            const childContainer = document.createElement('div');
            childContainer.className = 'children';
            children.forEach(child => child(childContainer));
            timelineEl.appendChild(childContainer);
        }}

        return row;
    }}

    function getOperatorColorClass(opType) {{
        const t = opType.toLowerCase();
        if (t.includes('scan')) return 'op-scan';
        if (t.includes('filter') || t.includes('project')) return 'op-filter';
        if (t.includes('join')) return 'op-join';
        if (t.includes('aggregat') || t.includes('agg')) return 'op-agg';
        if (t.includes('sort') || t.includes('order')) return 'op-sort';
        if (t.includes('exchange') || t.includes('partition') || t.includes('merge')) return 'op-exchange';
        if (t.includes('output') || t.includes('sink')) return 'op-output';
        return '';
    }}

    function createOperatorRow(operator, pipelineStartMs, pipelineEndMs, operatorStartMs, customLabel) {{
        // Operators have wall times (durations), not absolute timestamps
        // operatorStartMs is computed from cumulative times of previous operators
        const row = document.createElement('div');
        row.className = 'row operator';

        const labelEl = document.createElement('div');
        const colorClass = getOperatorColorClass(operator.operatorType);
        labelEl.className = 'label' + (colorClass ? ' ' + colorClass : '');
        labelEl.textContent = customLabel || (operator.operatorType + ' [' + operator.operatorId + ']');
        row.appendChild(labelEl);

        const barContainer = document.createElement('div');
        barContainer.className = 'bar-container';

        // Parse wall times
        const addInputMs = parseTimeString(operator.addInputWall);
        const getOutputMs = parseTimeString(operator.getOutputWall);
        const finishMs = parseTimeString(operator.finishWall);
        const blockedMs = parseTimeString(operator.blockedWall);
        const activeMs = addInputMs + getOutputMs + finishMs;  // Time actually doing work
        const totalMs = activeMs + blockedMs;

        if (operatorStartMs != null && totalMs > 0) {{
            let currentMs = operatorStartMs;

            const phases = [
                {{ name: 'AddInput', ms: addInputMs, cls: 'add-input', label: operator.addInputWall }},
                {{ name: 'GetOutput', ms: getOutputMs, cls: 'get-output', label: operator.getOutputWall }},
                {{ name: 'Finish', ms: finishMs, cls: 'finish', label: operator.finishWall }},
                {{ name: 'Blocked', ms: blockedMs, cls: 'blocked', label: operator.blockedWall }},
            ];

            phases.forEach(phase => {{
                if (phase.ms > 0) {{
                    const barStart = currentMs;
                    const barEnd = currentMs + phase.ms;
                    const bar = document.createElement('div');
                    bar.className = 'bar ' + phase.cls;
                    bar.style.left = toPercent(barStart) + '%';
                    bar.style.width = Math.max(0.3, toPercent(barEnd) - toPercent(barStart)) + '%';
                    bar.onmouseenter = (e) => showTooltip(e, operator.operatorType + ' - ' + phase.name, [
                        ['Operator ID', operator.operatorId],
                        ['Plan Node', operator.planNodeId || '-'],
                        [phase.name + ' Wall', phase.label || '-'],
                        ['Active Time', activeMs.toFixed(2) + 'ms'],
                        ['Blocked Time', blockedMs.toFixed(2) + 'ms'],
                        ['Total Wall', totalMs.toFixed(2) + 'ms'],
                    ]);
                    bar.onmousemove = (e) => {{ tooltipEl.style.left = (e.clientX + 15) + 'px'; tooltipEl.style.top = (e.clientY + 15) + 'px'; }};
                    bar.onmouseleave = hideTooltip;
                    barContainer.appendChild(bar);
                    currentMs = barEnd;
                }}
            }});
        }}

        row.appendChild(barContainer);
        timelineEl.appendChild(row);
        return row;
    }}

    function getOperatorTotalMs(operator) {{
        const addInputMs = parseTimeString(operator.addInputWall);
        const getOutputMs = parseTimeString(operator.getOutputWall);
        const finishMs = parseTimeString(operator.finishWall);
        const blockedMs = parseTimeString(operator.blockedWall);
        return addInputMs + getOutputMs + finishMs + blockedMs;
    }}

    function createPipelineHeader(label, tooltipData) {{
        const row = document.createElement('div');
        row.className = 'row pipeline';

        const labelEl = document.createElement('div');
        labelEl.className = 'label';
        labelEl.textContent = label;
        labelEl.style.cursor = 'pointer';
        labelEl.onmouseenter = (e) => showTooltip(e, tooltipData.title, tooltipData.rows);
        labelEl.onmousemove = (e) => {{ tooltipEl.style.left = (e.clientX + 15) + 'px'; tooltipEl.style.top = (e.clientY + 15) + 'px'; }};
        labelEl.onmouseleave = hideTooltip;
        row.appendChild(labelEl);

        // No bar container for pipeline header - just the label
        timelineEl.appendChild(row);
        return row;
    }}

    // Render stages
    data.stages.forEach(stage => {{
        const stageLabel = stage.stageId.split('.').pop();
        createRow('stage', 'Stage ' + stageLabel, stage.startTimeMs, stage.endTimeMs, {{
            title: 'Stage ' + stageLabel,
            rows: [
                ['State', stage.state || '-'],
                ['Start', stage.startTimeMs ? new Date(stage.startTimeMs).toISOString() : '-'],
                ['End', stage.endTimeMs ? new Date(stage.endTimeMs).toISOString() : '-'],
                ['Duration', stage.startTimeMs && stage.endTimeMs ? (stage.endTimeMs - stage.startTimeMs) + 'ms' : '-'],
                ['CPU Time', stage.totalCpuTime || '-'],
                ['Scheduled', stage.totalScheduledTime || '-'],
                ['Blocked', stage.totalBlockedTime || '-'],
            ]
        }});

        (stage.workers || []).forEach(worker => {{
            // Calculate worker time range from its tasks
            let wStart = Infinity, wEnd = -Infinity;
            (worker.tasks || []).forEach(t => {{
                if (t.createTimeMs) wStart = Math.min(wStart, t.createTimeMs);
                if (t.endTimeMs) wEnd = Math.max(wEnd, t.endTimeMs);
            }});
            if (wStart === Infinity) wStart = null;
            if (wEnd === -Infinity) wEnd = null;

            createRow('worker', worker.workerId, wStart, wEnd, {{
                title: 'Worker: ' + worker.workerId,
                rows: [
                    ['URI', worker.workerUri || '-'],
                    ['Tasks', (worker.tasks || []).length],
                    ['Start', wStart ? new Date(wStart).toISOString() : '-'],
                    ['End', wEnd ? new Date(wEnd).toISOString() : '-'],
                ]
            }});

            (worker.tasks || []).forEach(task => {{
                const taskLabel = task.taskId.split('.').slice(-2).join('.');
                createRow('task', 'Task ' + taskLabel, task.createTimeMs, task.endTimeMs, {{
                    title: 'Task ' + taskLabel,
                    rows: [
                        ['State', task.state || '-'],
                        ['Created', task.createTimeMs ? new Date(task.createTimeMs).toISOString() : '-'],
                        ['Started', task.firstStartTimeMs ? new Date(task.firstStartTimeMs).toISOString() : '-'],
                        ['Ended', task.endTimeMs ? new Date(task.endTimeMs).toISOString() : '-'],
                        ['Elapsed', formatNanos(task.elapsedTimeNanos)],
                        ['Queued', formatNanos(task.queuedTimeNanos)],
                        ['CPU', formatNanos(task.totalCpuTimeNanos)],
                        ['Scheduled', formatNanos(task.totalScheduledTimeNanos)],
                    ]
                }});

                // Build plan tree from pipelines with nesting based on data flow
                const pipelines = task.pipelines || [];
                const pipelineMap = {{}};

                // Index pipelines and find relationships
                pipelines.forEach(p => {{
                    const ops = p.operators || [];
                    pipelineMap[p.pipelineId] = {{
                        pipeline: p,
                        ops: ops,
                        firstNode: ops[0]?.planNodeId,
                        lastNode: ops[ops.length - 1]?.planNodeId,
                        lastType: ops[ops.length - 1]?.operatorType || '',
                        children: [],  // pipelines that feed into this one
                        parent: null,
                        insertAfterOp: null  // which operator this feeds into
                    }};
                }});

                // Find parent-child relationships
                // A pipeline feeds into another if:
                // 1. Its last node (LocalPartition) matches another's first node (LocalExchangeSink)
                // 2. Its HashJoinBuild matches another's HashJoinProbe node
                Object.values(pipelineMap).forEach(pInfo => {{
                    const lastNode = pInfo.lastNode;
                    const lastType = pInfo.lastType;
                    if (!lastNode) return;

                    Object.values(pipelineMap).forEach(otherInfo => {{
                        if (pInfo.pipeline.pipelineId === otherInfo.pipeline.pipelineId) return;

                        otherInfo.ops.forEach((op, opIdx) => {{
                            const opNode = op.planNodeId;
                            if (!opNode) return;

                            // LocalPartition -> LocalExchangeSink connection
                            if (lastType.includes('LocalPartition') && opNode === lastNode) {{
                                pInfo.parent = otherInfo.pipeline.pipelineId;
                                pInfo.insertAfterOp = opIdx;
                                otherInfo.children.push(pInfo.pipeline.pipelineId);
                            }}
                            // HashJoinBuild -> HashJoinProbe connection (same node ID)
                            else if (lastType.includes('Build') && op.operatorType?.includes('Probe') && lastNode === opNode) {{
                                pInfo.parent = otherInfo.pipeline.pipelineId;
                                pInfo.insertAfterOp = opIdx;
                                otherInfo.children.push(pInfo.pipeline.pipelineId);
                            }}
                        }});
                    }});
                }});

                // Find root pipelines (no parent)
                const rootPipelines = Object.values(pipelineMap)
                    .filter(p => p.parent === null)
                    .sort((a, b) => a.pipeline.pipelineId - b.pipeline.pipelineId);

                // Recursive function to render operators from a pipeline and its children
                // indent: string of spaces for current indentation level
                // This is now dataflow-oriented: no pipeline headers, just operators
                function renderPipelineTree(pInfo, indent, isLast) {{
                    const pipeline = pInfo.pipeline;
                    const ops = pInfo.ops;
                    if (ops.length === 0) return;

                    // Get child pipelines grouped by the operator they feed into
                    const childrenByOp = {{}};
                    pInfo.children.forEach(childId => {{
                        const childInfo = pipelineMap[childId];
                        const opIdx = childInfo.insertAfterOp || 0;
                        if (!childrenByOp[opIdx]) childrenByOp[opIdx] = [];
                        childrenByOp[opIdx].push(childInfo);
                    }});

                    // Render operators with nested child pipelines
                    let operatorStartMs = pipeline.firstStartTimeMs;

                    ops.forEach((operator, opIdx) => {{
                        const childrenAtOp = childrenByOp[opIdx] || [];
                        const isLastInPipeline = opIdx === ops.length - 1;
                        const hasMoreAfter = !isLastInPipeline || childrenAtOp.length > 0;

                        // Use isLast for the last operator of a last pipeline branch
                        const isLastOp = isLastInPipeline && isLast && childrenAtOp.length === 0;
                        const opPrefix = indent + (isLastOp ? '└─' : '├─');

                        // Shorten operator names
                        let opName = operator.operatorType
                            .replace(/Operator$/, '')
                            .replace(/^Cudf/, '');

                        const nodeId = operator.planNodeId || '';
                        const opLabel = opPrefix + opName + (nodeId ? ' [' + nodeId + ']' : '');

                        createOperatorRow(operator, pipeline.firstStartTimeMs, pipeline.lastEndTimeMs, operatorStartMs, opLabel);
                        operatorStartMs += getOperatorTotalMs(operator);

                        // Render child pipelines that feed into this operator
                        if (childrenAtOp.length > 0) {{
                            const opChildIndent = indent + (isLastOp ? '  ' : '│ ');
                            childrenAtOp.forEach((childInfo, childIdx) => {{
                                const isLastChild = childIdx === childrenAtOp.length - 1 && isLastInPipeline && isLast;
                                renderPipelineTree(childInfo, opChildIndent, isLastChild);
                            }});
                        }}
                    }});
                }}

                // Render from root pipelines
                rootPipelines.forEach((pInfo, idx) => {{
                    renderPipelineTree(pInfo, '', idx === rootPipelines.length - 1);
                }});
            }});
        }});
    }});

    // Render GPU charts if nsys data is available
    if (nsysData) {{
        const gpuChartsEl = document.getElementById('gpu-charts');

        // Convert nsys relative time (ms from session start) to absolute time (ms since epoch)
        // nsysData.sessionStartUtcNs is nanoseconds since epoch
        const nsysSessionStartMs = nsysData.sessionStartUtcNs / 1000000;
        const nsysToAbsolute = (relativeMs) => nsysSessionStartMs + relativeMs;

        function formatBytes(bytes) {{
            if (bytes < 1024) return bytes + ' B';
            if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
            if (bytes < 1024 * 1024 * 1024) return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
            return (bytes / (1024 * 1024 * 1024)).toFixed(2) + ' GB';
        }}

        function createChart(title, id) {{
            const chart = document.createElement('div');
            chart.className = 'gpu-chart';
            chart.innerHTML = '<h3>' + title + '</h3><div class="chart-container"><div class="chart-y-axis" id="' + id + '-yaxis"></div><div class="chart-area" id="' + id + '"></div></div>';
            gpuChartsEl.appendChild(chart);
            return {{
                area: document.getElementById(id),
                yaxis: document.getElementById(id + '-yaxis')
            }};
        }}

        function renderBarChart(chart, data, valueKey, barClass, maxVal) {{
            if (!data || data.length === 0) return;

            const area = chart.area;
            const yaxis = chart.yaxis;

            // Use the same time scale as the main timeline
            data.forEach(d => {{
                const absTimeMs = nsysToAbsolute(d.timeMs);
                if (absTimeMs < minTime || absTimeMs > maxTime) return;
                const val = d[valueKey] || 0;
                const pct = toPercent(absTimeMs);
                const height = maxVal > 0 ? (val / maxVal * 100) : 0;
                const bucketWidth = nsysData.bucketSizeMs / duration * 100;

                const bar = document.createElement('div');
                bar.className = 'chart-bar ' + barClass;
                bar.style.left = pct + '%';
                bar.style.width = Math.max(0.5, bucketWidth) + '%';
                bar.style.height = height + '%';
                bar.onmouseenter = (e) => showTooltip(e, 'Time: ' + d.timeMs.toFixed(0) + 'ms (relative)', [
                    ['Value', valueKey === 'throughputGBps' ? d[valueKey].toFixed(2) + ' GB/s' : formatBytes(d[valueKey])],
                    ['Count', d.count || '-'],
                ]);
                bar.onmousemove = (e) => {{ tooltipEl.style.left = (e.clientX + 15) + 'px'; tooltipEl.style.top = (e.clientY + 15) + 'px'; }};
                bar.onmouseleave = hideTooltip;
                area.appendChild(bar);
            }});

            // Y-axis labels
            yaxis.innerHTML = '<div>' + (valueKey === 'throughputGBps' ? maxVal.toFixed(1) + ' GB/s' : formatBytes(maxVal)) + '</div><div>0</div>';
        }}

        function renderLineChart(chart, data, valueKey, lineClass, maxVal) {{
            if (!data || data.length === 0) return;

            const area = chart.area;
            const yaxis = chart.yaxis;

            let prevX = null;
            let prevY = null;

            data.forEach(d => {{
                const absTimeMs = nsysToAbsolute(d.timeMs);
                if (absTimeMs < minTime || absTimeMs > maxTime) return;
                const val = d[valueKey] || 0;
                const x = toPercent(absTimeMs);
                const y = maxVal > 0 ? (val / maxVal * 100) : 0;

                if (prevX !== null) {{
                    // Draw a line segment
                    const line = document.createElement('div');
                    line.className = 'chart-line ' + lineClass;
                    line.style.left = prevX + '%';
                    line.style.width = (x - prevX) + '%';
                    line.style.bottom = prevY + '%';
                    area.appendChild(line);
                }}

                // Add a hoverable point
                const point = document.createElement('div');
                point.style.cssText = 'position:absolute;width:8px;height:8px;border-radius:50%;background:#9c27b0;transform:translate(-50%,-50%);cursor:pointer;z-index:1;';
                point.style.left = x + '%';
                point.style.bottom = y + '%';
                point.onmouseenter = (e) => showTooltip(e, 'Time: ' + d.timeMs.toFixed(0) + 'ms (relative)', [
                    ['Allocated', formatBytes(val)],
                ]);
                point.onmousemove = (e) => {{ tooltipEl.style.left = (e.clientX + 15) + 'px'; tooltipEl.style.top = (e.clientY + 15) + 'px'; }};
                point.onmouseleave = hideTooltip;
                area.appendChild(point);

                prevX = x;
                prevY = y;
            }});

            yaxis.innerHTML = '<div>' + formatBytes(maxVal) + '</div><div>0</div>';
        }}

        function renderPoolChart(chart, data, maxVal) {{
            if (!data || data.length === 0) return;

            const area = chart.area;
            const yaxis = chart.yaxis;

            let prevSizeX = null, prevSizeY = null;
            let prevUtilX = null, prevUtilY = null;

            data.forEach(d => {{
                const absTimeMs = nsysToAbsolute(d.timeMs);
                if (absTimeMs < minTime || absTimeMs > maxTime) return;
                const x = toPercent(absTimeMs);
                const sizeY = maxVal > 0 ? (d.poolSize / maxVal * 100) : 0;
                const utilY = maxVal > 0 ? (d.poolUtilized / maxVal * 100) : 0;

                // Pool size line (blue)
                if (prevSizeX !== null) {{
                    const line = document.createElement('div');
                    line.style.cssText = 'position:absolute;border-top:2px solid #2196f3;';
                    line.style.left = prevSizeX + '%';
                    line.style.width = (x - prevSizeX) + '%';
                    line.style.bottom = prevSizeY + '%';
                    area.appendChild(line);
                }}

                // Pool utilized line (green)
                if (prevUtilX !== null) {{
                    const line = document.createElement('div');
                    line.style.cssText = 'position:absolute;border-top:2px solid #4caf50;';
                    line.style.left = prevUtilX + '%';
                    line.style.width = (x - prevUtilX) + '%';
                    line.style.bottom = prevUtilY + '%';
                    area.appendChild(line);
                }}

                // Hoverable point
                const point = document.createElement('div');
                point.style.cssText = 'position:absolute;width:8px;height:8px;border-radius:50%;background:#4caf50;transform:translate(-50%,-50%);cursor:pointer;z-index:1;';
                point.style.left = x + '%';
                point.style.bottom = utilY + '%';
                point.onmouseenter = (e) => showTooltip(e, 'Time: ' + d.timeMs.toFixed(0) + 'ms (relative)', [
                    ['Pool Size', formatBytes(d.poolSize)],
                    ['Pool Utilized', formatBytes(d.poolUtilized)],
                    ['Utilization', ((d.poolUtilized / d.poolSize) * 100).toFixed(1) + '%'],
                ]);
                point.onmousemove = (e) => {{ tooltipEl.style.left = (e.clientX + 15) + 'px'; tooltipEl.style.top = (e.clientY + 15) + 'px'; }};
                point.onmouseleave = hideTooltip;
                area.appendChild(point);

                prevSizeX = x; prevSizeY = sizeY;
                prevUtilX = x; prevUtilY = utilY;
            }});

            yaxis.innerHTML = '<div>' + formatBytes(maxVal) + '</div><div>0</div>';
        }}

        // Calculate max values for scaling
        const maxHtod = Math.max(...(nsysData.htodThroughput || []).map(d => d.throughputGBps || 0), 0.1);
        const maxDtoh = Math.max(...(nsysData.dtohThroughput || []).map(d => d.throughputGBps || 0), 0.1);
        const maxThroughput = Math.max(maxHtod, maxDtoh);
        const maxMemory = Math.max(...(nsysData.cumulativeMemory || []).map(d => d.bytesAllocated || 0), 1);
        const maxPool = Math.max(...(nsysData.memoryPool || []).map(d => d.poolSize || 0), 1);

        // Create and render charts
        const htodChart = createChart('Host → Device Throughput', 'htod-chart');
        renderBarChart(htodChart, nsysData.htodThroughput, 'throughputGBps', 'htod', maxThroughput);

        const dtohChart = createChart('Device → Host Throughput', 'dtoh-chart');
        renderBarChart(dtohChart, nsysData.dtohThroughput, 'throughputGBps', 'dtoh', maxThroughput);

        const memChart = createChart('GPU Memory Allocated (Cumulative)', 'memory-chart');
        renderLineChart(memChart, nsysData.cumulativeMemory, 'bytesAllocated', 'memory', maxMemory);

        const poolChart = createChart('Local Memory Pool (Size: blue, Utilized: green)', 'pool-chart');
        renderPoolChart(poolChart, nsysData.memoryPool, maxPool);
    }}
    </script>
</body>
</html>
'''


def generate_timeline(summary_path: Path, output_path: Path | None = None, nsys_path: Path | None = None) -> Path:
    """Generate an HTML timeline from a summary.json file."""
    with open(summary_path) as f:
        summary = json.load(f)

    nsys_data = None
    if nsys_path and nsys_path.exists():
        with open(nsys_path) as f:
            nsys_data = json.load(f)

    if output_path is None:
        output_path = summary_path.parent / "timeline.html"

    html = HTML_TEMPLATE.format(
        query_id=summary.get("queryId", "unknown"),
        query_text=summary.get("query", "")[:500],
        state=summary.get("state", "-"),
        elapsed_time=summary.get("elapsedTime", "-"),
        execution_time=summary.get("executionTime", "-"),
        queued_time=summary.get("queuedTime", "-"),
        data_json=json.dumps(summary),
        nsys_json=json.dumps(nsys_data) if nsys_data else "null",
    )

    with open(output_path, "w") as f:
        f.write(html)

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Generate timeline HTML from summary.json")
    parser.add_argument("summary_json", type=Path, help="Path to summary.json file")
    parser.add_argument("-o", "--output", type=Path, help="Output HTML file path")
    parser.add_argument("--nsys", type=Path, help="Path to nsys_metrics.json file")
    args = parser.parse_args()

    output = generate_timeline(args.summary_json, args.output, args.nsys)
    print(f"Timeline generated: {output}")


if __name__ == "__main__":
    main()
