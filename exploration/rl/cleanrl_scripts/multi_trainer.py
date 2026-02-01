import argparse
import subprocess
import traceback
import signal
import sys
import time
from datetime import datetime
import pytz

sys.path.append('.')

from exploration.mdp.graph.problem_set import ProblemSet


class MultiTrainer:
    def __init__(self, problems, agent_load_paths, replay_buffer_files_path, hindsight_sharing, name, seed):
        israel_tz = pytz.timezone('Israel')
        now_in_israel = datetime.now(israel_tz)
        self.name = name or now_in_israel.strftime("%d-%m-%Y_%H-%M")
        self.seed = seed
        self.problems = problems
        self.agent_load_paths = agent_load_paths
        self.replay_buffer_files_path = replay_buffer_files_path
        self.hindsight_sharing = hindsight_sharing
        self.processes = []
        self.base_command = ['python', 'exploration/rl/cleanrl_scripts/sac_algorithm.py', '-n', self.name]

    def run(self):
        try:
            # Start all processes
            for idx, problem in enumerate(self.problems):
                command = self.base_command + ["-p", problem]
                if self.agent_load_paths is not None and len(self.agent_load_paths) > idx:
                    command += ["-alp", self.agent_load_paths[idx]]
                if self.replay_buffer_files_path is not None:
                    command += ["-rbp", self.replay_buffer_files_path]
                if self.hindsight_sharing is not None:
                    command += ["-hs", str(self.hindsight_sharing)]
                if self.seed is not None:
                    command += ["-s", str(self.seed)]
                process = subprocess.Popen(command)
                self.processes.append(process)
                print(f"Started process for {problem} with PID {process.pid}")

            # Monitor processes
            while self.processes:
                for i, p in enumerate(self.processes[:]):
                    # Check if process has terminated
                    return_code = p.poll()
                    if return_code is not None:
                        # Process has finished
                        if return_code != 0:
                            # Process failed
                            # stderr_output = p.stderr.read()
                            stderr_output = ''
                            raise Exception(f"Process for {self.problems[i]} failed with return code {return_code}.\nError output:\n{stderr_output}")

                        # Process completed successfully, remove from list
                        self.processes.remove(p)

                time.sleep(10)  # Short sleep to prevent CPU hogging

        except Exception as e:
            # Kill all remaining processes if something goes wrong
            self.terminate_all_processes()
            # Print traceback of the current exception
            traceback.print_exception(type(e), e, e.__traceback__)
            sys.exit(1)

    def terminate_all_processes(self):
        """Terminate all running child processes."""
        print("Terminating all processes due to an error...")
        for p in self.processes:
            if p.poll() is None:  # If process is still running
                try:
                    # Try graceful termination first
                    p.terminate()
                    # Give process a moment to terminate gracefully
                    time.sleep(5)
                    # Force kill if still running
                    if p.poll() is None:
                        p.kill()
                except Exception as kill_error:
                    print(f"Error terminating process: {kill_error}")


if __name__ == "__main__":
    # Set up a signal handler for cleaner termination
    def signal_handler(sig, frame):
        print(f"Received signal {sig}. Terminating all processes...")
        if 'level_trainer' in locals():
            level_trainer.terminate_all_processes()
        sys.exit(1)

    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # termination request

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--problems", type=str, nargs='+', default=None, help="Problems to use")
    parser.add_argument("-alp", "--agent_load_paths", type=str, nargs='+', default=None, help="Agent load paths to use")
    parser.add_argument("-rbp", "--replay_buffer_files_path", type=str, nargs='+', default=None, help="Replay buffer paths to use")
    parser.add_argument("-hs", "--hindsight_sharing", type=int, default=None, help="use hindsight sharing")
    parser.add_argument("-n", "--name", type=str, default=None, help="Name of the experiment")
    parser.add_argument("-s", "--seed", type=int, default=None, help="Seed for the experiment")
    args = parser.parse_args()

    problem_set = ProblemSet()
    problems = args.problems
    not_found_problems = []
    for problem_name in problems:
        if problem_name not in problem_set.PROBLEMS:
            not_found_problems.append(problem_name)
    assert len(not_found_problems) == 0, f"Problems not found: {not_found_problems}"

    level_trainer = MultiTrainer(
        problems=problems,
        agent_load_paths=args.agent_load_paths,
        replay_buffer_files_path=args.replay_buffer_files_path,
        hindsight_sharing=args.hindsight_sharing,
        name=args.name,
        seed=args.seed
    )

    try:
        level_trainer.run()
    except Exception as e:
        traceback.print_exception(type(e), e, e.__traceback__)
        sys.exit(1)
