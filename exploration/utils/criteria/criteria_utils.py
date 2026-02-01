import numpy as np
from typing import List, Tuple, Optional
from datetime import datetime

from scipy.stats import entropy

from exploration.mdp.istate import IState
from exploration.mdp.state_mapping import StateVisitation


def choose_by_criteria(states: List[IState], criteria_list: List[float], top_k: Optional[int] = None, total: int = 1) -> (List[IState], List[float]):
    criteria_list = np.array(criteria_list)
    criteria_unique = np.unique(criteria_list)[::-1]
    top_k = top_k if top_k is not None else len(criteria_unique)
    criteria_top_k = criteria_unique[:top_k]
    mask = np.in1d(criteria_list, criteria_top_k)
    criteria = criteria_list[mask]
    new_states = [state for state, is_true in zip(states, mask) if is_true]
    idxs = np.random.choice(len(criteria), size=total)
    states = [new_states[idx] for idx in idxs]
    cs = [criteria[idx] for idx in idxs]
    return states, cs


def calculate_criteria_on_state_visitation(state_visitation: StateVisitation[IState, int], criteria,
                                           sample_criteria: bool, sample_size: int = None,
                                           time_profiling: bool = False) -> Tuple[List[IState], List[float]]:
    st = datetime.now()

    states, criteria_list = [], []
    cache = {}
    num_new_visits = 0
    avg_time_new_visits = 0
    iterator = state_visitation.sample(size=sample_size).items() if sample_criteria else state_visitation.items()

    for state, visits in iterator:
        if visits in cache:  # TODO this is true for criteria which is dependent on visits only
            c = cache[visits]
        else:
            t0 = datetime.now()
            c = criteria.calculate(state=state, state_visitation=state_visitation)
            t1 = datetime.now()
            num_new_visits += 1
            num_new_visits, avg_time_new_visits = \
                calculate_time_stats(st=t0, et=t1, avg=avg_time_new_visits, total=num_new_visits)
            cache[visits] = c

        states.append(state)
        criteria_list.append(c)

    et = datetime.now()

    if time_profiling:
        print(f'CRITERIA CALCULATION - new visits called {num_new_visits} with avg time {avg_time_new_visits} ')
        print(f'CRITERIA CALCULATION - total: {et - st}')

    return states, criteria_list


def calculate_state_visitation_entropy(state_visitation: StateVisitation[IState, int],
                                       state_to_increase: Optional = None) -> float:
    if state_to_increase is not None:
        vs = np.array([v + 1 if s == state_to_increase else max(1, v) for s, v in state_visitation.items()])
        if state_to_increase not in state_visitation:
            vs = np.insert(vs, -1, 1)
        total = state_visitation.total + 1
    else:
        vs = np.maximum(1, np.array(list(state_visitation.values())))
        total = state_visitation.total

    probs = vs / total
    e = entropy(probs, base=2)

    return e


def calculate_iterative_mean(quantity: float, new_total: int, value: float):
    mean = ((new_total - 1) / new_total) * quantity + (1 / new_total) * value
    return mean


def calculate_time_stats(st: datetime, et: datetime, avg: float, total: int):
    total_sec = (et - st).total_seconds()
    if total == 0:
        avg = total_sec
    total += 1
    avg = calculate_iterative_mean(quantity=avg, new_total=total, value=total_sec)
    return total, avg
