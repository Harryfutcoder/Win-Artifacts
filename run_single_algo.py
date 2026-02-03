#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Single Algorithm Runner - Run one algorithm at a time, then merge results

This script runs each algorithm separately to avoid system crashes from long runs.
After all algorithms complete, merge results to generate final report and charts.

Usage:
    # Check current status
    python run_single_algo.py --status
    
    # Run single algorithm (3 runs, 1 hour each = ~3 hours)
    python run_single_algo.py --algo shaqv2
    python run_single_algo.py --algo shaq
    python run_single_algo.py --algo qtran
    python run_single_algo.py --algo vdn
    python run_single_algo.py --algo nndql
    
    # After all complete, merge results and generate report/charts
    python run_single_algo.py --merge
    
Algorithms: shaqv2, shaq, qtran, vdn, nndql
Each takes ~3 hours (1 hour x 3 runs)
"""

import subprocess
import json
import os
import sys
import argparse
from datetime import datetime
from pathlib import Path
import time
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

# Configuration
WORKSPACE = Path(r"c:\Users\SUST\Desktop\my-new-artifacts")
RESULTS_DIR = WORKSPACE / "ase_experiment_results" / "single_algo_results"
MERGED_OUTPUT = WORKSPACE / "ase_experiment_results" / "ase_rq1_merged.json"
FIGURE_DIR = WORKSPACE / "benchmark_figures"
REPORT_FILE = WORKSPACE / "ase_experiment_results" / "final_report.txt"
PYTHON_EXE = r"D:\Anaconda\python.exe"

ALGORITHMS = ['shaqv2', 'shaq', 'qtran', 'vdn', 'nndql', 'pshaq', 'jdpshaq']

# Test parameters
DURATION = 10800  # 3 hours per run
NUM_RUNS = 1      # 1 run per algorithm


def run_single_algorithm(algo: str):
    """Run a single algorithm experiment"""
    
    print("=" * 70)
    print(f"  RUNNING ALGORITHM: {algo.upper()}")
    print(f"  Duration: {DURATION}s x {NUM_RUNS} runs = {DURATION * NUM_RUNS / 3600:.1f} hours")
    print("=" * 70)
    
    # Create results directory
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Output file for this algorithm
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = RESULTS_DIR / f"{algo}_{timestamp}.json"
    log_file = RESULTS_DIR / f"{algo}_{timestamp}.log"
    
    # Build command
    cmd = [
        PYTHON_EXE, "benchmark.py",
        "--compare", algo,  # Single algorithm
        "--duration", str(DURATION),
        "--num-runs", str(NUM_RUNS),
        "--output", str(output_file),
        "--no-plot",  # Skip individual plots, generate at merge
    ]
    
    print(f"\nCommand: {' '.join(cmd)}")
    print(f"Output: {output_file}")
    print(f"Log: {log_file}")
    print()
    
    start_time = datetime.now()
    
    try:
        with open(log_file, 'w', encoding='utf-8') as log:
            log.write(f"Algorithm: {algo}\n")
            log.write(f"Start time: {start_time.isoformat()}\n")
            log.write(f"Command: {' '.join(cmd)}\n")
            log.write("=" * 70 + "\n\n")
            log.flush()
            
            # Run the benchmark
            process = subprocess.Popen(
                cmd,
                stdout=log,
                stderr=subprocess.STDOUT,
                cwd=str(WORKSPACE),
            )
            
            # Wait for completion
            process.wait()
            
            end_time = datetime.now()
            duration = end_time - start_time
            
            log.write("\n" + "=" * 70 + "\n")
            log.write(f"End time: {end_time.isoformat()}\n")
            log.write(f"Duration: {duration}\n")
            log.write(f"Exit code: {process.returncode}\n")
        
        if process.returncode == 0:
            print(f"\n{'='*70}")
            print(f"  [SUCCESS] {algo.upper()} COMPLETED!")
            print(f"  Duration: {duration}")
            print(f"  Results: {output_file}")
            print(f"{'='*70}")
            return True
        else:
            print(f"\n[FAILED] {algo} failed with exit code {process.returncode}")
            print(f"  Check log: {log_file}")
            return False
            
    except Exception as e:
        print(f"\n[ERROR] {algo} error: {e}")
        return False


def check_status():
    """Check which algorithms have completed"""
    
    print("=" * 70)
    print("  EXPERIMENT STATUS")
    print("=" * 70)
    
    if not RESULTS_DIR.exists():
        print("\nNo results found yet.")
        print("\nStart with: python run_single_algo.py --algo shaqv2")
        return
    
    completed = []
    pending = []
    
    for algo in ALGORITHMS:
        # Look for result files
        results = list(RESULTS_DIR.glob(f"{algo}_*.json"))
        if results:
            # Get latest result
            latest = max(results, key=lambda p: p.stat().st_mtime)
            size = latest.stat().st_size
            mtime = datetime.fromtimestamp(latest.stat().st_mtime)
            
            # Check if valid JSON with results
            try:
                with open(latest, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Check for aggregated results (means all 3 runs completed)
                if 'aggregated_results' in data and data['aggregated_results']:
                    agg = list(data['aggregated_results'].values())[0]
                    runs = agg.get('num_runs', 0)
                    states = agg.get('unique_states_mean', 0)
                    completed.append((algo, latest.name, runs, states, size, mtime))
                else:
                    pending.append(algo)
            except:
                pending.append(algo)
        else:
            pending.append(algo)
    
    print("\n[Completed Algorithms]")
    if completed:
        for algo, fname, runs, states, size, mtime in completed:
            print(f"  [OK] {algo:8s}: {runs} runs, {states:.0f} states, {size/1024:.1f}KB ({mtime.strftime('%m-%d %H:%M')})")
    else:
        print("  (none yet)")
    
    print("\n[Pending Algorithms]")
    if pending:
        for algo in pending:
            print(f"  [ ] {algo}")
    else:
        print("  (none - all complete!)")
    
    # Progress bar
    total = len(ALGORITHMS)
    done = len(completed)
    pct = done / total * 100
    bar = "#" * done + "-" * (total - done)
    print(f"\n  Progress: [{bar}] {done}/{total} ({pct:.0f}%)")
    
    print()
    if pending:
        print(f"  Next command: python run_single_algo.py --algo {pending[0]}")
    else:
        print("  All algorithms complete!")
        print("  Generate report: python run_single_algo.py --merge")


def generate_comparison_report(merged_data: Dict) -> str:
    """Generate text comparison report (same format as benchmark.py)"""
    
    agg_results = merged_data.get('aggregated_results', {})
    
    if not agg_results:
        return "No aggregated results available."
    
    lines = []
    lines.append("=" * 80)
    lines.append("  MULTI-AGENT REINFORCEMENT LEARNING ALGORITHM COMPARISON REPORT")
    lines.append("=" * 80)
    lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"  Algorithms: {', '.join(agg_results.keys())}")
    lines.append("")
    
    # Key metrics comparison
    metrics = [
        ('unique_states', 'State Coverage', 'count'),
        ('unique_urls', 'URL Coverage', 'count'),
        ('total_steps', 'Total Steps', 'steps'),
        ('effective_steps_per_second', 'Throughput', 'steps/s'),
        ('avg_decision_time_ms', 'Decision Time', 'ms'),
        ('total_reward', 'Total Reward', 'points'),
        ('exploration_efficiency', 'Exploration Efficiency', 'ratio'),
    ]
    
    lines.append("-" * 80)
    lines.append("  KEY METRICS COMPARISON (mean ± std)")
    lines.append("-" * 80)
    
    for metric_key, metric_name, unit in metrics:
        lines.append(f"\n  {metric_name} ({unit}):")
        
        values = []
        for algo, data in agg_results.items():
            # Handle nested metrics structure: data['metrics'][metric_key]['mean/std']
            if 'metrics' in data and metric_key in data['metrics']:
                mean = data['metrics'][metric_key].get('mean', 0)
                std = data['metrics'][metric_key].get('std', 0)
            else:
                # Fallback to flat structure
                mean = data.get(f'{metric_key}_mean', 0)
                std = data.get(f'{metric_key}_std', 0)
            values.append((algo, mean, std))
        
        # For decision time, lower is better (ascending sort)
        # For other metrics, higher is better (descending sort)
        lower_is_better = 'time' in metric_key.lower()
        values.sort(key=lambda x: x[1], reverse=not lower_is_better)
        
        # Check if all values are the same (no winner)
        unique_vals = set(v[1] for v in values)
        has_winner = len(unique_vals) > 1
        
        for i, (algo, mean, std) in enumerate(values):
            marker = "★" if i == 0 and has_winner else " "
            if mean >= 1000:
                lines.append(f"    {marker} {algo:10s}: {mean:>10,.0f} ± {std:>8,.0f}")
            elif mean >= 1:
                lines.append(f"    {marker} {algo:10s}: {mean:>10.2f} ± {std:>8.2f}")
            else:
                lines.append(f"    {marker} {algo:10s}: {mean:>10.4f} ± {std:>8.4f}")
    
    # Winner summary
    lines.append("\n" + "=" * 80)
    lines.append("  WINNER SUMMARY")
    lines.append("=" * 80)
    
    winners = {}
    for metric_key, metric_name, _ in metrics:
        best_algo = None
        best_val = -float('inf')
        all_values = []
        
        # Collect all values first
        for algo, data in agg_results.items():
            if 'metrics' in data and metric_key in data['metrics']:
                val = data['metrics'][metric_key].get('mean', 0)
            else:
                val = data.get(f'{metric_key}_mean', 0)
            all_values.append((algo, val))
        
        # Check if all values are the same (no real winner)
        unique_values = set(v for _, v in all_values)
        if len(unique_values) <= 1:
            # All values are the same, skip this metric
            continue
        
        # For decision time, lower is better
        if 'time' in metric_key.lower():
            best_val = float('inf')
            for algo, val in all_values:
                if val < best_val:
                    best_val = val
                    best_algo = algo
        else:
            for algo, val in all_values:
                if val > best_val:
                    best_val = val
                    best_algo = algo
        
        if best_algo:
            if best_algo not in winners:
                winners[best_algo] = []
            winners[best_algo].append(metric_name)
    
    for algo in agg_results.keys():
        won = winners.get(algo, [])
        if won:
            lines.append(f"  {algo:10s}: Won {len(won)} metrics - {', '.join(won)}")
        else:
            lines.append(f"  {algo:10s}: (no wins)")
    
    lines.append("\n" + "=" * 80)
    
    return "\n".join(lines)


def merge_results():
    """Merge all single algorithm results into one file and generate report/charts"""
    
    print("=" * 70)
    print("  MERGING RESULTS & GENERATING REPORT")
    print("=" * 70)
    
    if not RESULTS_DIR.exists():
        print("\nNo results found!")
        return False
    
    merged_single = {}
    merged_aggregated = {}
    
    print("\n[Loading Results]")
    for algo in ALGORITHMS:
        results = list(RESULTS_DIR.glob(f"{algo}_*.json"))
        if not results:
            print(f"  [MISSING] {algo} - no results found")
            continue
        
        # Get latest result
        latest = max(results, key=lambda p: p.stat().st_mtime)
        print(f"  [FOUND] {algo}: {latest.name}")
        
        try:
            with open(latest, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract single run results
            if 'single_run_results' in data:
                for profile, runs in data['single_run_results'].items():
                    merged_single[profile] = runs
            
            # Extract aggregated results
            if 'aggregated_results' in data:
                for profile, agg in data['aggregated_results'].items():
                    merged_aggregated[profile] = agg
                    
        except Exception as e:
            print(f"  [ERROR] {algo}: {e}")
    
    if not merged_single:
        print("\nNo valid results to merge!")
        return False
    
    # Create merged output
    merged_data = {
        'metadata': {
            'merged_at': datetime.now().isoformat(),
            'algorithms': list(merged_aggregated.keys()),
            'source_dir': str(RESULTS_DIR),
            'num_runs_per_algo': NUM_RUNS,
            'duration_per_run': DURATION,
        },
        'single_run_results': merged_single,
        'aggregated_results': merged_aggregated,
    }
    
    # Ensure output directory exists
    MERGED_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    
    # Save merged results
    with open(MERGED_OUTPUT, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n[Merged JSON] {MERGED_OUTPUT}")
    
    # Generate text report
    print("\n[Generating Report]")
    report = generate_comparison_report(merged_data)
    print(report)
    
    with open(REPORT_FILE, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n[Report Saved] {REPORT_FILE}")
    
    # Generate comparison charts
    print("\n[Generating Charts]")
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    
    try:
        from benchmark import BenchmarkVisualizer
        visualizer = BenchmarkVisualizer(output_dir=str(FIGURE_DIR))
        if visualizer.available:
            visualizer.plot_comparison(str(MERGED_OUTPUT))
            
            # Also generate confidence interval chart if we have aggregated data
            if merged_aggregated:
                visualizer.plot_reward_with_confidence(merged_aggregated)
            
            print(f"  [OK] Charts saved to: {FIGURE_DIR}")
            
            # List generated files
            charts = list(FIGURE_DIR.glob("*.png"))
            for chart in charts:
                print(f"    - {chart.name}")
        else:
            print("  [WARN] matplotlib not available, skipping charts")
    except Exception as e:
        print(f"  [ERROR] Chart generation failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("  MERGE COMPLETE!")
    print("=" * 70)
    print(f"\n  Results JSON: {MERGED_OUTPUT}")
    print(f"  Report TXT:   {REPORT_FILE}")
    print(f"  Charts DIR:   {FIGURE_DIR}")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Run algorithms one at a time, then merge results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check current status
  python run_single_algo.py --status
  
  # Run one algorithm (~3 hours)
  python run_single_algo.py --algo shaqv2
  
  # After all complete, merge and generate report/charts
  python run_single_algo.py --merge

Order: shaqv2 -> shaq -> qtran -> vdn -> nndql
Each takes ~3 hours (1 hour x 3 runs)
Total: ~15 hours (but can be done across multiple sessions)
"""
    )
    
    parser.add_argument('--algo', choices=ALGORITHMS,
                       help='Algorithm to run')
    parser.add_argument('--merge', action='store_true',
                       help='Merge all completed results and generate report/charts')
    parser.add_argument('--status', action='store_true',
                       help='Check status of all algorithms')
    
    args = parser.parse_args()
    
    if args.status:
        check_status()
    elif args.merge:
        merge_results()
    elif args.algo:
        # Disable sleep before running
        print("\n[System] Disabling sleep mode...")
        os.system("powercfg /change standby-timeout-ac 0")
        os.system("powercfg /change hibernate-timeout-ac 0")
        os.system("powercfg /change monitor-timeout-ac 0")
        print("[System] Sleep mode disabled\n")
        
        success = run_single_algorithm(args.algo)
        
        if success:
            # Show next steps
            check_status()
    else:
        check_status()


if __name__ == '__main__':
    main()
