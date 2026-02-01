from system_flow.metrics.h_values import H_VALUES_4_CROSSES, H_VALUES_3_CROSSES

EASY_STATES = [
    '1: U 4 -\n2: U 5 +\n3: O 6 +\n4: O 1 -\n5: O 2 +\n6: U 3 +',
    '1: O 6 +\n2: U 5 +\n3: U 4 -\n4: O 3 -\n5: O 2 +\n6: U 1 +',
    '1: U 4 +\n2: U 5 -\n3: O 6 -\n4: O 1 +\n5: O 2 -\n6: U 3 -',
    '1: O 6 -\n2: U 5 -\n3: U 4 +\n4: O 3 +\n5: O 2 -\n6: U 1 -',
    '1: U 6 +\n2: U 5 -\n3: O 4 -\n4: U 3 -\n5: O 2 -\n6: O 1 +',
    '1: U 6 -\n2: U 5 +\n3: U 4 -\n4: O 3 -\n5: O 2 +\n6: O 1 -',
    '1: U 6 -\n2: O 5 -\n3: O 4 +\n4: U 3 +\n5: U 2 -\n6: O 1 -',
    '1: O 6 +\n2: U 5 +\n3: O 4 +\n4: U 3 +\n5: O 2 +\n6: U 1 +',
    '1: O 6 -\n2: U 5 -\n3: O 4 -\n4: U 3 -\n5: O 2 -\n6: U 1 -',
    '1: U 6 -\n2: O 5 -\n3: U 4 -\n4: O 3 -\n5: U 2 -\n6: O 1 -',
]

MEDIUM_STATES = [
    '1: U 6 +\n2: U 5 +\n3: U 4 -\n4: O 3 -\n5: O 2 +\n6: O 1 +',
    '1: U 3 -\n2: O 6 -\n3: O 1 -\n4: O 5 +\n5: U 4 +\n6: U 2 -',
    '1: O 6 +\n2: O 3 -\n3: U 2 -\n4: O 5 +\n5: U 4 +\n6: U 1 +',
    '1: O 3 +\n2: U 4 +\n3: U 1 +\n4: O 2 +\n5: O 6 -\n6: U 5 -',
    '1: O 2 +\n2: U 1 +\n3: O 6 -\n4: U 5 +\n5: O 4 +\n6: U 3 -',
    '1: U 2 -\n2: O 1 -\n3: U 4 +\n4: O 3 +\n5: O 6 -\n6: U 5 -',
    '1: O 2 +\n2: U 1 +\n3: O 5 -\n4: O 6 +\n5: U 3 -\n6: U 4 +',
    '1: O 5 -\n2: O 3 -\n3: U 2 -\n4: O 6 +\n5: U 1 -\n6: U 4 +',
    '1: U 2 -\n2: O 1 -\n3: O 4 +\n4: U 3 +\n5: O 6 -\n6: U 5 -',
    '1: U 3 -\n2: O 6 -\n3: O 1 -\n4: U 5 -\n5: O 4 -\n6: U 2 -',
]

HARD_STATES = [
    '1: O 2 +\n2: U 1 +\n3: O 5 +\n4: U 6 +\n5: U 3 +\n6: O 4 +',
    '1: O 3 +\n2: U 6 +\n3: U 1 +\n4: O 5 +\n5: U 4 +\n6: O 2 +',
    '1: U 2 -\n2: O 1 -\n3: O 6 -\n4: U 5 +\n5: O 4 +\n6: U 3 -',
    '1: O 4 -\n2: U 3 +\n3: O 2 +\n4: U 1 -\n5: U 6 -\n6: O 5 -',
    '1: O 2 -\n2: U 1 -\n3: U 5 +\n4: O 6 +\n5: O 3 +\n6: U 4 +',
    '1: U 3 +\n2: O 4 +\n3: O 1 +\n4: U 2 +\n5: U 6 +\n6: O 5 +',
    '1: U 4 +\n2: O 3 -\n3: U 2 -\n4: O 1 +\n5: U 6 -\n6: O 5 -',
    '1: U 6 -\n2: O 3 +\n3: U 2 +\n4: O 5 +\n5: U 4 +\n6: O 1 -',
    '1: O 2 +\n2: U 1 +\n3: U 5 -\n4: O 6 -\n5: O 3 -\n6: U 4 -',
    '1: U 4 -\n2: O 5 -\n3: U 6 -\n4: O 1 -\n5: U 2 -\n6: O 3 -',
]

COMPLEXITY_STATES = {
    "easy": EASY_STATES,
    "medium": MEDIUM_STATES,
    "hard": HARD_STATES
}

EVAL_3_RANDOM_BINS_STATES = [
    '1: O 3 +\n2: O 6 -\n3: U 1 +\n4: U 5 -\n5: O 4 -\n6: U 2 -',
    '1: O 3 -\n2: U 6 -\n3: U 1 -\n4: O 5 -\n5: U 4 -\n6: O 2 -',
    '1: O 3 +\n2: O 4 -\n3: U 1 +\n4: U 2 -\n5: O 6 -\n6: U 5 -',
    '1: U 3 +\n2: O 6 +\n3: O 1 +\n4: U 5 -\n5: O 4 -\n6: U 2 +',
    '1: O 3 -\n2: O 4 +\n3: U 1 -\n4: U 2 +\n5: O 6 -\n6: U 5 -',
    '1: U 6 -\n2: U 3 -\n3: O 2 -\n4: O 5 +\n5: U 4 +\n6: O 1 -',
    '1: O 2 -\n2: U 1 -\n3: O 4 -\n4: U 3 -\n5: U 6 -\n6: O 5 -',
    '1: U 2 +\n2: O 1 +\n3: U 4 +\n4: O 3 +\n5: O 6 +\n6: U 5 +',
    '1: O 2 +\n2: U 1 +\n3: U 4 -\n4: O 3 -\n5: O 6 +\n6: U 5 +',
    '1: U 2 +\n2: O 1 +\n3: U 4 -\n4: O 3 -\n5: O 6 -\n6: U 5 -',
    '1: U 2 +\n2: O 1 +\n3: U 4 -\n4: O 3 -\n5: O 6 +\n6: U 5 +',
    '1: O 2 -\n2: U 1 -\n3: O 4 +\n4: U 3 +\n5: O 6 -\n6: U 5 -',
    '1: O 5 +\n2: U 6 +\n3: O 4 -\n4: U 3 -\n5: U 1 +\n6: O 2 +',
    '1: U 6 +\n2: U 3 +\n3: O 2 +\n4: O 5 +\n5: U 4 +\n6: O 1 +',
    '1: U 2 +\n2: O 1 +\n3: O 4 -\n4: U 3 -\n5: O 6 -\n6: U 5 -',
    '1: O 2 -\n2: U 1 -\n3: U 4 -\n4: O 3 -\n5: O 6 +\n6: U 5 +',
    '1: O 2 -\n2: U 1 -\n3: O 5 +\n4: U 6 +\n5: U 3 +\n6: O 4 +',
    '1: O 2 +\n2: U 1 +\n3: U 4 +\n4: O 3 +\n5: U 6 +\n6: O 5 +',
    '1: U 2 -\n2: O 1 -\n3: O 5 +\n4: O 6 -\n5: U 3 +\n6: U 4 -',
    '1: O 2 +\n2: U 1 +\n3: O 4 +\n4: U 3 +\n5: O 6 +\n6: U 5 +',
    '1: O 2 +\n2: U 1 +\n3: O 4 +\n4: U 3 +\n5: U 6 +\n6: O 5 +',
    '1: U 6 -\n2: U 5 +\n3: O 4 -\n4: U 3 -\n5: O 2 +\n6: O 1 -',
    '1: O 2 +\n2: U 1 +\n3: O 5 -\n4: O 6 +\n5: U 3 -\n6: U 4 +',
    '1: U 6 +\n2: U 5 +\n3: U 4 -\n4: O 3 -\n5: O 2 +\n6: O 1 +',
    '1: U 2 -\n2: O 1 -\n3: O 4 +\n4: U 3 +\n5: O 6 -\n6: U 5 -',
    '1: U 2 +\n2: O 1 +\n3: U 5 -\n4: O 6 -\n5: O 3 -\n6: U 4 -',
    '1: O 2 -\n2: U 1 -\n3: O 4 -\n4: U 3 -\n5: O 6 +\n6: U 5 +',
    '1: O 2 -\n2: U 1 -\n3: U 6 -\n4: U 5 -\n5: O 4 -\n6: O 3 -',
    '1: O 4 -\n2: O 6 -\n3: U 5 -\n4: U 1 -\n5: O 3 -\n6: U 2 -',
    '1: U 3 -\n2: U 4 +\n3: O 1 -\n4: O 2 +\n5: O 6 +\n6: U 5 +',
]

EVAL_4_EASY_800 = [
    '1: U 6 +\n2: U 5 -\n3: O 8 +\n4: O 7 -\n5: O 2 -\n6: O 1 +\n7: U 4 -\n8: U 3 +',
    '1: U 6 -\n2: U 5 +\n3: O 8 -\n4: O 7 +\n5: O 2 +\n6: O 1 -\n7: U 4 +\n8: U 3 -',
    '1: O 4 +\n2: U 5 +\n3: O 8 +\n4: U 1 +\n5: O 2 +\n6: U 7 -\n7: O 6 -\n8: U 3 +',
    '1: U 7 -\n2: O 3 +\n3: U 2 +\n4: O 8 -\n5: U 6 +\n6: O 5 +\n7: O 1 -\n8: U 4 -',
    '1: U 7 -\n2: U 3 -\n3: O 2 -\n4: O 8 -\n5: U 6 +\n6: O 5 +\n7: O 1 -\n8: U 4 -',
    '1: U 3 +\n2: O 6 +\n3: O 1 +\n4: U 5 +\n5: O 4 +\n6: U 2 +\n7: U 8 -\n8: O 7 -',
    '1: O 3 -\n2: U 4 -\n3: U 1 -\n4: O 2 -\n5: O 8 +\n6: U 7 +\n7: O 6 +\n8: U 5 +',
    '1: U 6 -\n2: U 3 -\n3: O 2 -\n4: U 7 +\n5: O 8 +\n6: O 1 -\n7: O 4 +\n8: U 5 +',
    '1: U 6 -\n2: U 7 +\n3: O 4 +\n4: U 3 +\n5: O 8 +\n6: O 1 -\n7: O 2 +\n8: U 5 +',
    '1: U 5 +\n2: O 6 +\n3: U 8 -\n4: O 7 -\n5: O 1 +\n6: U 2 +\n7: U 4 -\n8: O 3 -',
]


if __name__ == '__main__':
    easy_h_values = [H_VALUES_3_CROSSES[k] for k in EASY_STATES]
    print(f"{EASY_STATES=} {easy_h_values=}")
    medium_h_values = [H_VALUES_3_CROSSES[k] for k in MEDIUM_STATES]
    print(f"{MEDIUM_STATES=} {medium_h_values=}")
    hard_h_values = [H_VALUES_3_CROSSES[k] for k in HARD_STATES]
    print(f"{HARD_STATES=} {hard_h_values=}")
    eval_3_random_bins_h_values = [H_VALUES_3_CROSSES[k] for k in EVAL_3_RANDOM_BINS_STATES]
    print(f"{EVAL_3_RANDOM_BINS_STATES=} {eval_3_random_bins_h_values=}")
    eval_4_easy_800_h_values = [H_VALUES_4_CROSSES[k] for k in EVAL_4_EASY_800]
    print(f"{EVAL_4_EASY_800=} {eval_4_easy_800_h_values=}")