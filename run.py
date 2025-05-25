from pathlib import Path
import subprocess
import sys
import time
from threading import Thread


BASE_DIR = Path(__file__).parent
APP_LOC = BASE_DIR / "app" / "app.py"
API_LOC = BASE_DIR / "api" / "api.py"
PYTHON = BASE_DIR / ".mentat" / "bin" / "python"


def run_process(command, process_name):
    """Run a process and print its output"""
    print(f"Starting {process_name}...")
    try:
        # Use shell=True for command strings or shell=False for command lists
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            shell=isinstance(command, str)
        )

        # Keep track of processes for clean shutdown
        running_processes.append(process)

        # Print output in real-time
        for line in process.stdout:
            print(f"[{process_name}] {line.strip()}")

        process.wait()
        print(f"{process_name} exited with code {process.returncode}")

    except Exception as e:
        print(f"Error in {process_name}: {e}")


def handle_shutdown(*args):
    """Handle termination signals by shutting down all processes"""
    print("\nShutting down processes...")
    for process in running_processes:
        if process.poll() is None:  # If process is still running
            process.terminate()
            # Give it a moment to terminate gracefully
            time.sleep(0.5)
            if process.poll() is None:  # If still running after terminate
                process.kill()  # Force kill

    print("All processes terminated")
    sys.exit(0)


if __name__ == "__main__":

    import os
    import signal

    # Append project root to sys.path
    sys.path.append(BASE_DIR.as_posix())

    # Register signal handlers for clean shutdown
    signal.signal(signal.SIGINT, handle_shutdown)  # Ctrl+C
    signal.signal(signal.SIGTERM, handle_shutdown)  # Termination

    # List to track running processes for clean shutdown
    running_processes = []

    # Create threads to run each process
    RUN_APP = f"{PYTHON} {APP_LOC}"
    RUN_API = f"{PYTHON} {API_LOC}"
    thread1 = Thread(target=run_process, args=(RUN_APP, "App/UI"))
    thread2 = Thread(target=run_process, args=(RUN_API, "API/Server"))

    # Start both processes
    thread1.start()
    thread2.start()

    try:
        # Wait for both to complete
        thread1.join()
        thread2.join()
    except KeyboardInterrupt:
        # This will be caught by the signal handler
        pass

    print("All processes have completed.")