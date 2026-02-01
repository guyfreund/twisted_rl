import numpy as np
from collections import OrderedDict

from system_flow.metrics.h_values import H_VALUES_3_CROSSES
from system_flow.metrics.states import EVAL_3_RANDOM_BINS_STATES, EVAL_3_RANDOM_BINS_STATES_OLD

seed = 37

np.random.seed(seed)

highest_count = 150
bin_size = 10
states_in_each_bins = 2

bins = highest_count // bin_size
total = bins * states_in_each_bins

h_values_bins = {}
for i in range(1, bins + 1):
    lower, upper = (i - 1) * bin_size + 1, i * bin_size
    h_values = {state_idx: counts for state_idx, counts in H_VALUES_3_CROSSES.items() if lower < counts <= upper}
    h_values_sorted = OrderedDict(sorted(h_values.items(), key=lambda item: item[1]))
    h_values_bins[(lower, upper)] = h_values_sorted

state_idxs, counts = [], []
for (lower, upper), h_values in h_values_bins.items():
    sampled_idxs = np.random.choice(list(h_values.keys()), states_in_each_bins, replace=False).tolist()
    sampled_counts = [h_values[state_idx] for state_idx in sampled_idxs]
    state_idxs.extend(sampled_idxs)
    counts.extend(sampled_counts)
    print(f"Sampled H values idxs for {(lower, upper)}: {sampled_idxs=} {sampled_counts=}")

print()
sorted_idx = np.argsort(counts)
state_idxs = np.array(state_idxs)[sorted_idx].tolist()
counts = np.array(counts)[sorted_idx].tolist()
print(f"Sampled states: {state_idxs=}")
print(f"Sampled counts: {counts=}")

assert set(state_idxs) == set(EVAL_3_RANDOM_BINS_STATES), "Sampled states do not match the expected set of states."