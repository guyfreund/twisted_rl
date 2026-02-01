from dataclasses import dataclass
from typing import Optional


@dataclass
class EvaluationSet:
    name: str
    easy_path: str
    medium_path: str
    medium_1h_path: str
    hard_path: str
    four_easy_eval_path: str
    three_eval_path: str

    def get_path_for_state_type(self, state_type: str) -> Optional[str]:
        """Get the path for a specific state type."""
        if state_type == '3-Easy':
            return self.easy_path
        elif state_type == '3-Medium':
            return self.medium_path
        elif state_type == '3-Medium-1h':
            return self.medium_1h_path
        elif state_type == '3-Hard':
            return self.hard_path
        elif state_type in ['4-Easy-Eval', '4-Eval']:
            return self.four_easy_eval_path
        elif state_type == '3-Eval':
            return self.three_eval_path
        else:
            raise ValueError(f"Unknown state type: {state_type}")

    def paths(self) -> list[str]:
        """Return all paths in a list."""
        return [
            self.easy_path,
            self.medium_path,
            self.medium_1h_path,
            self.hard_path,
            self.four_easy_eval_path,
            self.three_eval_path
        ]


TWISTED = EvaluationSet(
    name="TWISTED",
    easy_path="exploration/outputs/evaluation/twisted_evaluation/22-04-2025_20-08/",
    medium_path="exploration/outputs/evaluation/twisted_evaluation/19-04-2025_01-01/",
    medium_1h_path="exploration/outputs/evaluation/twisted_evaluation/22-04-2025_01-34/",
    hard_path="exploration/outputs/evaluation/twisted_evaluation/19-04-2025_20-45/",
    four_easy_eval_path="exploration/outputs/evaluation/twisted_evaluation/21-04-2025_01-15/",
    three_eval_path="exploration/outputs/evaluation/twisted_evaluation/22-04-2025_16-36/"
)

TWISTED_RL_G = EvaluationSet(
    name="TWISTED-RL-G",
    easy_path="exploration/outputs/evaluation/twisted_evaluation/23-04-2025_17-13/",
    medium_path="exploration/outputs/evaluation/twisted_evaluation/23-04-2025_17-05/",
    medium_1h_path="exploration/outputs/evaluation/twisted_evaluation/26-04-2025_17-52/",
    hard_path="exploration/outputs/evaluation/twisted_evaluation/23-04-2025_17-11/",
    four_easy_eval_path="exploration/outputs/evaluation/twisted_evaluation/23-04-2025_17-14/",
    three_eval_path="exploration/outputs/evaluation/twisted_evaluation/23-04-2025_17-12/"
)

TWISTED_RL_A = EvaluationSet(
    name="TWISTED-RL-A",
    easy_path="exploration/outputs/evaluation/twisted_evaluation/26-04-2025_22-54/",
    medium_path="exploration/outputs/evaluation/twisted_evaluation/26-04-2025_22-55/",
    medium_1h_path="exploration/outputs/evaluation/twisted_evaluation/26-04-2025_22-56/",
    hard_path="exploration/outputs/evaluation/twisted_evaluation/26-04-2025_22-57/",
    four_easy_eval_path="exploration/outputs/evaluation/twisted_evaluation/26-04-2025_22-58/",
    three_eval_path="exploration/outputs/evaluation/twisted_evaluation/26-04-2025_22-59/"
)

TWISTED_RL_C = EvaluationSet(
    name="TWISTED-RL-C",
    easy_path="exploration/outputs/evaluation/twisted_evaluation/03-05-2025_11-07/",
    medium_path="exploration/outputs/evaluation/twisted_evaluation/03-05-2025_11-08/",
    medium_1h_path="exploration/outputs/evaluation/twisted_evaluation/04-05-2025_10-27/",
    hard_path="exploration/outputs/evaluation/twisted_evaluation/03-05-2025_13-45/",
    four_easy_eval_path="exploration/outputs/evaluation/twisted_evaluation/06-05-2025_00-43/",
    three_eval_path="exploration/outputs/evaluation/twisted_evaluation/04-05-2025_10-29/"
)

TWISTED_RL_AC = EvaluationSet(
    name="TWISTED-RL-AC",
    easy_path="exploration/outputs/evaluation/twisted_evaluation/22-04-2025_20-09/",
    medium_path="exploration/outputs/evaluation/twisted_evaluation/19-04-2025_01-00/",
    medium_1h_path="exploration/outputs/evaluation/twisted_evaluation/22-04-2025_01-35/",
    hard_path="exploration/outputs/evaluation/twisted_evaluation/19-04-2025_20-43/",
    four_easy_eval_path="exploration/outputs/evaluation/twisted_evaluation/21-04-2025_01-18/",
    three_eval_path="exploration/outputs/evaluation/twisted_evaluation/24-04-2025_18-47/"
)

TWISTED_RL_H = EvaluationSet(
    name="TWISTED-RL-H",
    easy_path="exploration/outputs/evaluation/twisted_evaluation/10-05-2025_18-42/",
    medium_path="exploration/outputs/evaluation/twisted_evaluation/10-05-2025_18-44/",
    medium_1h_path="exploration/outputs/evaluation/twisted_evaluation/11-05-2025_22-38",
    hard_path="exploration/outputs/evaluation/twisted_evaluation/10-05-2025_23-54/",
    four_easy_eval_path="exploration/outputs/evaluation/twisted_evaluation/11-05-2025_22-39",
    three_eval_path="exploration/outputs/evaluation/twisted_evaluation/11-05-2025_22-42"
)

TWISTED_RL_ALL = EvaluationSet(  # Full
    name="TWISTED-RL-ALL",
    easy_path="exploration/outputs/evaluation/twisted_evaluation/14-05-2025_14-51/",
    medium_path="exploration/outputs/evaluation/twisted_evaluation/14-05-2025_14-50/",
    medium_1h_path="exploration/outputs/evaluation/twisted_evaluation/16-05-2025_11-31/",
    hard_path="exploration/outputs/evaluation/twisted_evaluation/14-05-2025_20-29/",
    four_easy_eval_path="exploration/outputs/evaluation/twisted_evaluation/15-05-2025_11-10/",
    three_eval_path="exploration/outputs/evaluation/twisted_evaluation/16-05-2025_00-42/"
)

TWISTED_RL_ALL_G = EvaluationSet(  # G Removal
    name="TWISTED-RL-ALL-G",
    easy_path="exploration/outputs/evaluation/twisted_evaluation/23-05-2025_12-11/",
    medium_path="exploration/outputs/evaluation/twisted_evaluation/23-05-2025_12-12/",
    medium_1h_path="exploration/outputs/evaluation/twisted_evaluation/25-05-2025_17-15/",
    hard_path="exploration/outputs/evaluation/twisted_evaluation/23-05-2025_16-19/",
    four_easy_eval_path="exploration/outputs/evaluation/twisted_evaluation/24-05-2025_15-26/",
    three_eval_path="exploration/outputs/evaluation/twisted_evaluation/24-05-2025_15-28/",
)

TWISTED_RL_ALL_A = EvaluationSet(  # A Removal
    name="TWISTED-RL-ALL-A",
    easy_path="exploration/outputs/evaluation/twisted_evaluation/27-05-2025_17-46/",
    medium_path="exploration/outputs/evaluation/twisted_evaluation/27-05-2025_17-44/",
    medium_1h_path="exploration/outputs/evaluation/twisted_evaluation/29-05-2025_18-18/",
    hard_path="exploration/outputs/evaluation/twisted_evaluation/28-05-2025_13-42/",
    four_easy_eval_path="exploration/outputs/evaluation/twisted_evaluation/28-05-2025_13-44/",
    three_eval_path="exploration/outputs/evaluation/twisted_evaluation/29-05-2025_18-17/",
)

TWISTED_RL_ALL_C = EvaluationSet(  # C Removal
    name="TWISTED-RL-ALL-C",
    easy_path="exploration/outputs/evaluation/twisted_evaluation/31-05-2025_12-46/",
    medium_path="exploration/outputs/evaluation/twisted_evaluation/31-05-2025_21-48/",
    medium_1h_path="exploration/outputs/evaluation/twisted_evaluation/03-06-2025_11-33/",
    hard_path="exploration/outputs/evaluation/twisted_evaluation/01-06-2025_12-38/",
    four_easy_eval_path="exploration/outputs/evaluation/twisted_evaluation/02-06-2025_12-47/",
    three_eval_path="exploration/outputs/evaluation/twisted_evaluation/02-06-2025_12-48/",
)

TWISTED_RL_ALL_AC = EvaluationSet(  # AC Removal
    name="TWISTED-RL-ALL-AC",
    easy_path="exploration/outputs/evaluation/twisted_evaluation/05-06-2025_09-38/",
    medium_path="exploration/outputs/evaluation/twisted_evaluation/05-06-2025_09-39/",
    medium_1h_path="exploration/outputs/evaluation/twisted_evaluation/07-06-2025_09-06/",
    hard_path="exploration/outputs/evaluation/twisted_evaluation/05-06-2025_14-07/",
    four_easy_eval_path="exploration/outputs/evaluation/twisted_evaluation/06-06-2025_10-07/",
    three_eval_path="exploration/outputs/evaluation/twisted_evaluation/06-06-2025_15-39/",
)

TWISTED_RL_C5S = EvaluationSet(
    name="TWISTED-RL-C5S",
    easy_path="exploration/outputs/evaluation/twisted_evaluation/09-06-2025_13-20/",
    medium_path="exploration/outputs/evaluation/twisted_evaluation/09-06-2025_13-24/",
    medium_1h_path="exploration/outputs/evaluation/twisted_evaluation/11-06-2025_13-59/",
    hard_path="exploration/outputs/evaluation/twisted_evaluation/10-06-2025_00-35/",
    four_easy_eval_path="exploration/outputs/evaluation/twisted_evaluation/10-06-2025_13-25/",
    three_eval_path="exploration/outputs/evaluation/twisted_evaluation/11-06-2025_13-57/",
)

TWISTED_RL_C5 = EvaluationSet(
    name="TWISTED-RL-C5",
    easy_path="exploration/outputs/evaluation/twisted_evaluation/13-06-2025_18-13/",
    medium_path="exploration/outputs/evaluation/twisted_evaluation/13-06-2025_18-25/",
    medium_1h_path="exploration/outputs/evaluation/twisted_evaluation/15-06-2025_13-34/",
    hard_path="exploration/outputs/evaluation/twisted_evaluation/14-06-2025_10-57/",
    four_easy_eval_path="exploration/outputs/evaluation/twisted_evaluation/14-06-2025_15-36/",
    three_eval_path="exploration/outputs/evaluation/twisted_evaluation/15-06-2025_13-32/",
)

TWISTED_RL_G5 = EvaluationSet(
    name="TWISTED-RL-G5",
    easy_path="exploration/outputs/evaluation/twisted_evaluation/18-07-2025_14-37/",
    medium_path="exploration/outputs/evaluation/twisted_evaluation/18-07-2025_17-52/",
    medium_1h_path="exploration/outputs/evaluation/twisted_evaluation/19-07-2025_20-41/",
    hard_path="exploration/outputs/evaluation/twisted_evaluation/21-07-2025_13-29/",
    four_easy_eval_path="exploration/outputs/evaluation/twisted_evaluation/22-07-2025_13-55/",
    three_eval_path="exploration/outputs/evaluation/twisted_evaluation/19-07-2025_20-40/",
)

TWISTED_RL_A5 = EvaluationSet(
    name="TWISTED-RL-A5",
    easy_path="exploration/outputs/evaluation/twisted_evaluation/14-07-2025_12-03/",
    medium_path="exploration/outputs/evaluation/twisted_evaluation/14-07-2025_12-08/",
    medium_1h_path="exploration/outputs/evaluation/twisted_evaluation/17-07-2025_10-05/",
    hard_path="exploration/outputs/evaluation/twisted_evaluation/14-07-2025_16-50/",
    four_easy_eval_path="exploration/outputs/evaluation/twisted_evaluation/15-07-2025_18-29/",
    three_eval_path="exploration/outputs/evaluation/twisted_evaluation/15-07-2025_18-27/",
)

TWISTED_RL_AC5 = EvaluationSet(
    name="TWISTED-RL-AC5",
    easy_path="exploration/outputs/evaluation/twisted_evaluation/22-07-2025_18-03/",
    medium_path="exploration/outputs/evaluation/twisted_evaluation/22-07-2025_22-22/",
    medium_1h_path="exploration/outputs/evaluation/twisted_evaluation/25-07-2025_15-30/",
    hard_path="exploration/outputs/evaluation/twisted_evaluation/23-07-2025_13-15/",
    four_easy_eval_path="exploration/outputs/evaluation/twisted_evaluation/24-07-2025_14-34/",
    three_eval_path="exploration/outputs/evaluation/twisted_evaluation/24-07-2025_13-26/",
)

TWISTED5 = EvaluationSet(
    name="TWISTED5",
    easy_path="exploration/outputs/evaluation/twisted_evaluation/27-07-2025_17-46/",
    medium_path="exploration/outputs/evaluation/twisted_evaluation/27-07-2025_17-47/",
    medium_1h_path="exploration/outputs/evaluation/twisted_evaluation/27-07-2025_17-48/",
    hard_path="exploration/outputs/evaluation/twisted_evaluation/27-07-2025_17-49/",
    four_easy_eval_path="exploration/outputs/evaluation/twisted_evaluation/27-07-2025_17-50/",
    three_eval_path="exploration/outputs/evaluation/twisted_evaluation/27-07-2025_17-51/",
)

TWISTED_RL_H5 = EvaluationSet(
    name="TWISTED-RL-H5",
    easy_path="exploration/outputs/evaluation/twisted_evaluation//",
    medium_path="exploration/outputs/evaluation/twisted_evaluation/02-08-2025_14-30/",
    medium_1h_path="exploration/outputs/evaluation/twisted_evaluation/02-08-2025_14-26/",
    hard_path="exploration/outputs/evaluation/twisted_evaluation/02-08-2025_14-29/",
    four_easy_eval_path="exploration/outputs/evaluation/twisted_evaluation/02-08-2025_14-31/",
    three_eval_path="exploration/outputs/evaluation/twisted_evaluation/02-08-2025_14-22/",
)

AVAILABLE_ABLATIONS = [
    TWISTED,
    TWISTED_RL_G,
    TWISTED_RL_A,
    TWISTED_RL_C,
    TWISTED_RL_AC,
    TWISTED_RL_H,
    TWISTED_RL_ALL,
    TWISTED_RL_ALL_G,
    TWISTED_RL_ALL_A,
    TWISTED_RL_ALL_C,
    TWISTED_RL_ALL_AC,
    TWISTED_RL_C5S,
    TWISTED_RL_G5,
    TWISTED_RL_A5,
    TWISTED_RL_C5,
    TWISTED_RL_AC5,
    TWISTED5,
    TWISTED_RL_H5,
]
