
# conda create -n sk python=3.13 psutil matplotlib-base seaborn -y
# pip install signifikante

import os
import time
import threading
import psutil
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.transforms as mtransforms

from typing import Tuple, List, Dict, Callable, Any, Union
from signifikante.fdr_utils import compute_wasserstein_distance_matrix

matplotlib.use('Agg')


def get_cpu_memory_mb(process: psutil.Process) -> float:
    total_mem = 0
    try:
        with process.oneshot():
            children = process.children(recursive=True)
            all_procs = [process] + children
            for proc in all_procs:

                try:
                    if proc.is_running():
                        total_mem += proc.memory_info().rss

                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

    except Exception as e:
        print(f'CPU memory tracking failed with error:\n{e}')

    total_mem /= 1024 ** 2

    return total_mem


def track_memory_cpu(interval: float):
    """
    Tracks total memory (RSS) of the current process + children.
    Returns a list of memory samples (in MB).
    """

    process = psutil.Process(os.getpid())
    memory_samples = [get_cpu_memory_mb(process=process)]
    stop_event = threading.Event()

    # Initial sample
    def poll():
        while not stop_event.is_set():
            mem = get_cpu_memory_mb(process=process)
            memory_samples.append(mem)
            stop_event.wait(interval)

    thread = threading.Thread(target=poll, daemon=True)
    thread.start()

    return memory_samples, stop_event, thread


def scalability_wrapper(
        function: Callable,
        function_params: Union[Dict[str, Any], None]= None,
        tracking_interval: float = 0.1,
) -> Tuple[float, List[float], Any]:

    # Start memory tracking
    memory_samples_cpu, stop_event_cpu, tracker_thread_cpu = track_memory_cpu(interval=tracking_interval)

    # Start timing
    wall_start = time.perf_counter()

    function_output = None
    try:

        if function_params is not None:
            function_output = function(**function_params)
        else:
            function_output = function()

    finally:

        wall_end = time.perf_counter()

        # Stop memory tracker
        stop_event_cpu.set()
        tracker_thread_cpu.join()

    # Analyze results
    wall_time = wall_end - wall_start

    return wall_time, memory_samples_cpu, function_output


def generate_results():

    np.random.seed(42)

    # Warmup run
    data_mat = pd.DataFrame(data=np.random.rand(10, 20), columns=[f'gene_{i}' for i in range(20)])
    compute_wasserstein_distance_matrix(expression_mat=data_mat, num_threads=-1)

    # Define params
    num_cells = [1000, 5000, 10000]
    num_genes = [1000, 2000, 5000, 10000, 20000, 30000]
    num_trials = 10

    # Run benchmark
    rows = []
    for trial in range(num_trials):
        for nc in num_cells:
            for ng in num_genes:

                print(f'# --- trial: {trial}, num cells: {nc}, num genes: {ng} ---')

                # Generate random data matrix
                data_mat = pd.DataFrame(data=np.random.rand(nc, ng), columns=[f'gene_{i}' for i in range(ng)])

                def run():
                    compute_wasserstein_distance_matrix(expression_mat=data_mat, num_threads=-1)

                wt, mem_samples, _ = scalability_wrapper(function=run, tracking_interval=0.01)

                mem_peak = max(mem_samples)
                num_samples = len(mem_samples)
                mem_avg = sum(mem_samples) / num_samples

                rows.append({
                    'num_cells': nc,
                    'num_genes': ng,
                    'trial': trial,
                    'wall_time': wt,
                    'mem_peak': mem_peak,
                    'mem_avg': mem_avg,
                    'num_samples': num_samples,
                })

                res_df = pd.DataFrame(rows)
                res_df.to_csv('results.csv', index=False)

def plot():

    res_df = pd.read_csv('results.csv')

    res_df_agg = (
        res_df
        .groupby(['num_cells', 'num_genes'], as_index=False)
        .mean(numeric_only=True)
    )

    print('# Bulk:\n', res_df_agg.loc[(res_df_agg['num_cells'] == 1000) & (res_df_agg['num_genes'] == 20000), :])

    print('# Single cell:\n', res_df_agg.loc[(res_df_agg['num_cells'] == 10000) & (res_df_agg['num_genes'] == 2000), :])

    # Convert MiB into GiB
    res_df['mem_peak'] /= 1024

    fig, axd = plt.subplot_mosaic(
        '''
        AB
        ''',
        figsize=(6.5, 3),
        dpi=300,
        constrained_layout=True,
    )

    ax = axd['A']
    sns.lineplot(
        data=res_df,
        x='num_genes',
        y='wall_time',
        hue='num_cells',
        marker='o',
        errorbar=None,
        ax=ax,
    )
    ax.set_xlabel('Number of genes')
    ax.set_ylabel('Wall time [s]')
    ax.get_legend().set_title('Number of samples')

    ax = axd['B']
    sns.lineplot(
        data=res_df,
        x='num_genes',
        y='mem_peak',
        hue='num_cells',
        marker='o',
        errorbar=None,
        ax=ax,
    )
    ax.set_xlabel('Number of genes')
    ax.set_ylabel('Peak memory [GiB]')
    ax.get_legend().set_title('Number of samples')

    for label, ax in axd.items():
        trans = mtransforms.ScaledTranslation(-20 / 72, 7 / 72, fig.dpi_scale_trans)
        ax.text(
            - 0.05,
            0.92,
            label,
            transform=ax.transAxes + trans,
            fontsize=16,
            va='bottom',
            fontfamily='sans-serif',
            fontweight='bold'
        )

    fig.savefig('wasserstein_benchmark.pdf', dpi=300)


if __name__ == '__main__':

    # generate_results()

    # plot()

    print('done')
