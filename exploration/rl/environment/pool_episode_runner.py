import os
import multiprocessing
import queue
import psutil
from typing import Optional, List
import traceback

from exploration.rl.environment.exploration_worker_env import ExplorationWorkerEnv

mp_context = multiprocessing.get_context('spawn')


def _init(requests_queues_per_worker: List[multiprocessing.Queue], response_queue: multiprocessing.Queue, env_kwargs):
    from absl import logging
    logging.set_verbosity(logging.FATAL)
    mp_context.current_process().requests_queues_per_worker = requests_queues_per_worker
    mp_context.current_process().response_queue = response_queue
    mp_context.current_process().env = ExplorationWorkerEnv(**env_kwargs)
    # report all the process ids to the main process
    response_queue.put(os.getpid())


def _work(function_input):
    query_id, query, policy_uses_id, worker_pids = function_input
    current_process = mp_context.current_process()
    my_index = worker_pids.index(os.getpid())
    requests_queue = current_process.requests_queues_per_worker[my_index]
    response_queue = current_process.response_queue

    def _put_message(message):
        response_queue.put(message)

    def _get_action(current_state):
        if policy_uses_id:
            state_message = my_index, 0, (current_state, query_id)
        else:
            state_message = my_index, 0, current_state
        _put_message(state_message)
        action = requests_queue.get(block=True, timeout=None)
        return action

    env = mp_context.current_process().env
    env.reset(**query)
    try:
        episode_experiences = env.play_episode(get_action=_get_action)
    except Exception as e:
        traceback.print_exception(type(e), e, e.__traceback__)
        raise e
    _put_message((my_index, 1, (query_id, episode_experiences)))


def _get_from_response_queue(response_queue: multiprocessing.Queue):
    messages = [response_queue.get(block=True)]
    while True:
        try:
            message = response_queue.get(block=True, timeout=0.001)
            messages.append(message)
        except queue.Empty:
            break
    return messages


class PoolContext:
    def __init__(self, workers: Optional[int], **env_params):
        # set the number of workers
        max_workers = max(os.cpu_count() - 2, 1)
        if workers is None or workers < 1:
            workers = max_workers
        else:
            workers = min(max_workers, workers)
        self.workers = workers

        # create queues
        self.requests_queues_per_worker = [mp_context.Queue() for _ in range(workers)]
        self.response_queue = mp_context.Queue()

        self.initargs = (self.requests_queues_per_worker, self.response_queue, env_params)
        self.pool = mp_context.Pool(self.workers, initializer=_init, initargs=self.initargs)

        # get all the workers pids
        worker_pids = []
        while len(worker_pids) < self.workers:
            worker_pids.extend(_get_from_response_queue(self.response_queue))
        self.worker_pids = tuple(sorted(worker_pids))

    def close(self):
        print('pool exit called')
        self.pool.close()
        self.pool.join()


def run_queries_parallel(
        policy_func, post_processing_function, queries, pool_context: PoolContext, policy_uses_id=False
):
    function_inputs = [
        (query_id, query, policy_uses_id, pool_context.worker_pids) for query_id, query in enumerate(queries)
    ]
    execution_future = pool_context.pool.map_async(_work, function_inputs)

    results = []
    while len(results) < len(queries):
        # get pending messages
        messages = _get_from_response_queue(pool_context.response_queue)

        # process messages
        worker_ids, current_states, query_ids = [], [], []
        for message in messages:
            worker_id, message_type, message_content = message
            if message_type == 0:
                # add to policy inputs
                worker_ids.append(worker_id)
                if policy_uses_id:
                    current_states.append(message_content[0])
                    query_ids.append(message_content[1])
                else:
                    current_states.append(message_content)

            elif message_type == 1:
                # terminated episodes - add to results
                if post_processing_function is not None:
                    results.append(post_processing_function(message_content))
                else:
                    results.append(message_content)

        # get actions from policy
        if len(current_states) > 0:
            if policy_uses_id:
                actions, stddev = policy_func(current_states, query_ids)
            else:
                actions, stddev = policy_func(current_states)
            for i, (worker_id, action) in enumerate(zip(worker_ids, actions)):
                action_stddev = stddev[i] if stddev is not None else None
                pool_context.requests_queues_per_worker[worker_id].put((action, action_stddev))

    return results


def get_subprocess_count():
    current_process = psutil.Process()
    children = current_process.children(recursive=True)
    return len(children)


def is_daemon():
    from multiprocessing import current_process
    return current_process().daemon


def get_memory_usage_run():
    current_process = psutil.Process()
    if is_daemon():
        current_process = current_process.parent()
    total_memory_used = current_process.memory_info().rss
    children = current_process.children()
    total_memory_used += sum([child.memory_info().rss for child in children])
    return total_memory_used / psutil.virtual_memory().total


def get_memory_usage_machine():
    v_mem = psutil.virtual_memory()
    return v_mem.used / v_mem.total
