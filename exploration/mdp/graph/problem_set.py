from typing import Optional, Dict

from exploration.mdp.graph.high_level_graph import Problem, HighLevelGraph


class ProblemSet:
    def __init__(self):
        self.G_kwargs = {
            'depth': 3,
            'max_crosses': 3,
            'min_crosses': 0,
            'high_level_actions': ['R1', 'R2', 'cross'],
        }
        self.G = Problem.load_kwargs(
            path=Problem.base_path(),
            name='G',
            description='Full High-Level Graph up to 3 crossings',
            build_parallel=True,
            **self.G_kwargs,
        )

        self.G_R1_kwargs = {
            'depth': 3,
            'max_crosses': 3,
            'min_crosses': 0,
            'high_level_actions': ['R1'],
        }
        self.G_R1 = Problem.load_kwargs(
            path=Problem.base_path(),
            name='G_R1',
            description='Full High-Level Graph up to 3 crossings R1',
            build_parallel=True,
            **self.G_R1_kwargs,
        )

        self.G_R2_kwargs = {
            'depth': 3,
            'max_crosses': 3,
            'min_crosses': 0,
            'high_level_actions': ['R2'],
        }
        self.G_R2 = Problem.load_kwargs(
            path=Problem.base_path(),
            name='G_R2',
            description='Full High-Level Graph up to 3 crossings R2',
            build_parallel=True,
            **self.G_R2_kwargs,
        )

        self.G_Cross_kwargs = {
            'depth': 3,
            'max_crosses': 3,
            'min_crosses': 0,
            'high_level_actions': ['cross'],
        }
        self.G_Cross = Problem.load_kwargs(
            path=Problem.base_path(),
            name='G_Cross',
            description='Full High-Level Graph up to 3 crossings Cross',
            build_parallel=True,
            **self.G_Cross_kwargs,
        )

        self.G0_kwargs = {
            'depth': 1,
            'min_crosses': 0,
            'max_crosses': 2,
            'high_level_actions': ['R1', 'R2'],  # no cross in crossing number 0
        }
        self.G0 = Problem.load_kwargs(
            path=Problem.base_path(),
            name='G0',
            description='High-Level Bipartite Graph C=0',
            **self.G0_kwargs,
        )

        self.G0_R1_kwargs = {
            'depth': 1,
            'min_crosses': 0,
            'max_crosses': 1,
            'high_level_actions': ['R1'],
        }
        self.G0_R1 = Problem.load_kwargs(
            path=Problem.base_path(),
            name='G0_R1',
            description='High-Level Bipartite Graph C=0 R1',
            **self.G0_R1_kwargs,
        )

        self.G0_R2_kwargs = {
            'depth': 1,
            'min_crosses': 0,
            'max_crosses': 2,
            'high_level_actions': ['R2'],
        }
        self.G0_R2 = Problem.load_kwargs(
            path=Problem.base_path(),
            name='G0_R2',
            description='High-Level Bipartite Graph C=0 R2',
            **self.G0_R2_kwargs,
        )

        self.G1_kwargs = {
            'depth': 1,
            'min_crosses': 1,
            'max_crosses': 3,
            'high_level_actions': ['R1', 'R2', 'cross'],
        }
        self.G1 = Problem.load_kwargs(
            path=Problem.base_path(),
            name='G1',
            description='High-Level Bipartite Graph C=1',
            **self.G1_kwargs,
        )

        self.G1_R1_kwargs = {
            'depth': 1,
            'min_crosses': 1,
            'max_crosses': 2,
            'high_level_actions': ['R1'],
        }
        self.G1_R1 = Problem.load_kwargs(
            path=Problem.base_path(),
            name='G1_R1',
            description='High-Level Bipartite Graph C=1 R1',
            **self.G1_R1_kwargs,
        )

        self.G1_R2_kwargs = {
            'depth': 1,
            'min_crosses': 1,
            'max_crosses': 3,
            'high_level_actions': ['R2'],
        }
        self.G1_R2 = Problem.load_kwargs(
            path=Problem.base_path(),
            name='G1_R2',
            description='High-Level Bipartite Graph C=1 R2',
            **self.G1_R2_kwargs,
        )

        self.G1_Cross_kwargs = {
            'depth': 1,
            'min_crosses': 1,
            'max_crosses': 2,
            'high_level_actions': ['cross'],
        }
        self.G1_Cross = Problem.load_kwargs(
            path=Problem.base_path(),
            name='G1_Cross',
            description='High-Level Bipartite Graph C=1 Cross',
            **self.G1_Cross_kwargs,
        )

        self.G2_kwargs = {
            'depth': 1,
            'min_crosses': 2,
            'max_crosses': 4,
            'high_level_actions': ['R1', 'R2', 'cross'],
        }
        self.G2 = Problem.load_kwargs(
            path=Problem.base_path(),
            name='G2',
            description='High-Level Bipartite Graph C=2',
            **self.G2_kwargs,
        )

        self.G2_R1_kwargs = {
            'depth': 1,
            'min_crosses': 2,
            'max_crosses': 3,
            'high_level_actions': ['R1'],
        }
        self.G2_R1 = Problem.load_kwargs(
            path=Problem.base_path(),
            name='G2_R1',
            description='High-Level Bipartite Graph C=2 R1',
            **self.G2_R1_kwargs,
        )

        self.G2_R2_kwargs = {
            'depth': 1,
            'min_crosses': 2,
            'max_crosses': 4,
            'high_level_actions': ['R2'],
        }
        self.G2_R2 = Problem.load_kwargs(
            path=Problem.base_path(),
            name='G2_R2',
            description='High-Level Bipartite Graph C=2 R2',
            **self.G2_R2_kwargs,
        )

        self.G2_Cross_kwargs = {
            'depth': 1,
            'min_crosses': 2,
            'max_crosses': 3,
            'high_level_actions': ['cross'],
        }
        self.G2_Cross = Problem.load_kwargs(
            path=Problem.base_path(),
            name='G2_Cross',
            description='High-Level Bipartite Graph C=2 Cross',
            **self.G2_Cross_kwargs,
        )

        self.G3_kwargs = {
            'depth': 1,
            'min_crosses': 3,
            'max_crosses': 5,
            'high_level_actions': ['R1', 'R2', 'cross'],
        }
        self.G3 = Problem.load_kwargs(
            path=Problem.base_path(),
            name='G3',
            description='High-Level Bipartite Graph C=3',
            **self.G3_kwargs,
        )
        self.G3_R1_kwargs = {
            'depth': 1,
            'min_crosses': 3,
            'max_crosses': 4,
            'high_level_actions': ['R1'],
        }
        self.G3_R1 = Problem.load_kwargs(
            path=Problem.base_path(),
            name='G3_R1',
            description='High-Level Bipartite Graph C=3 R1',
            **self.G3_R1_kwargs,
        )

        self.G3_R2_kwargs = {
            'depth': 1,
            'min_crosses': 3,
            'max_crosses': 5,
            'high_level_actions': ['R2'],
        }
        self.G3_R2 = Problem.load_kwargs(
            path=Problem.base_path(),
            name='G3_R2',
            description='High-Level Bipartite Graph C=3 R2',
            **self.G3_R2_kwargs,
        )

        self.G3_Cross_kwargs = {
            'depth': 1,
            'min_crosses': 3,
            'max_crosses': 4,
            'high_level_actions': ['cross'],
        }
        self.G3_Cross = Problem.load_kwargs(
            path=Problem.base_path(),
            name='G3_Cross',
            description='High-Level Bipartite Graph C=3 Cross',
            **self.G3_Cross_kwargs,
        )

        self.PROBLEMS: Dict[str, Problem] = {
            self.G_R1.name: self.G_R1,
            self.G_R2.name: self.G_R2,
            self.G_Cross.name: self.G_Cross,
            self.G.name: self.G,
            self.G0_R1.name: self.G0_R1,
            self.G0_R2.name: self.G0_R2,
            self.G0.name: self.G0,
            self.G1_R1.name: self.G1_R1,
            self.G1_R2.name: self.G1_R2,
            self.G1_Cross.name: self.G1_Cross,
            self.G1.name: self.G1,
            self.G2_R1.name: self.G2_R1,
            self.G2_R2.name: self.G2_R2,
            self.G2_Cross.name: self.G2_Cross,
            self.G2.name: self.G2,
            self.G3_R1.name: self.G3_R1,
            self.G3_R2.name: self.G3_R2,
            self.G3_Cross.name: self.G3_Cross,
            self.G3.name: self.G3,
        }

        self.PROBLEM_TO_KWARGS: Dict[str, Dict] = {
            self.G_R1.name: self.G_R1_kwargs,
            self.G_R2.name: self.G_R2_kwargs,
            self.G_Cross.name: self.G_Cross_kwargs,
            self.G.name: self.G_kwargs,
            self.G0_R1.name: self.G0_R1_kwargs,
            self.G0_R2.name: self.G0_R2_kwargs,
            self.G0.name: self.G0_kwargs,
            self.G1_R1.name: self.G1_R1_kwargs,
            self.G1_R2.name: self.G1_R2_kwargs,
            self.G1_Cross.name: self.G1_Cross_kwargs,
            self.G1.name: self.G1_kwargs,
            self.G2_R1.name: self.G2_R1_kwargs,
            self.G2_R2.name: self.G2_R2_kwargs,
            self.G2_Cross.name: self.G2_Cross_kwargs,
            self.G2.name: self.G2_kwargs,
            self.G3_R1.name: self.G3_R1_kwargs,
            self.G3_R2.name: self.G3_R2_kwargs,
            self.G3_Cross.name: self.G3_Cross_kwargs,
            self.G3.name: self.G3_kwargs,
        }

    def get_problem_by_kwargs(self, **kwargs) -> Optional[Problem]:
        for problem_name, problem_kwargs in self.PROBLEM_TO_KWARGS.items():
            if problem_kwargs == kwargs:
                return self.PROBLEMS[problem_name]
        return None

    def get_problem_by_name(self, name: str) -> Optional[Problem]:
        return self.PROBLEMS.get(name)

    def print_set(self, states: bool, edges: bool):
        for _, problem in self.PROBLEMS.items():
            problem.print_problem(states=states, edges=edges)


if __name__ == '__main__':
    print_states = True
    print_edges = True
    problem_set = ProblemSet()
    problem_set.print_set(states=print_states, edges=print_edges)
    high_level_graph = HighLevelGraph.load_full_graph()
