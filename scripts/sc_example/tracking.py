
import os
import time
import threading
import psutil

from typing import Tuple, List, Dict, Callable, Any, Union


def get_cpu_memory_mb(process: psutil.Process) -> float:
    """
    Returns total RSS memory usage (in MB) of a process and all child processes.
    """

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
    Starts a background thread that periodically samples total process RSS memory usage.
    Returns memory samples, stop event, and tracking thread.
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
    """
    Measures wall-clock runtime and tracks CPU memory usage while executing a function.
    Returns runtime, memory samples, and function output.
    """

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

