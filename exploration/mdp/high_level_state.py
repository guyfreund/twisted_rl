from exploration.mdp.istate import IState
from mujoco_infra.mujoco_utils.topology.representation import AbstractState


class HighLevelAbstractState(AbstractState, IState):
    def __init__(self):
        AbstractState.__init__(self=self)

    @property
    def np_encoding(self):
        return None

    @property
    def torch_encoding(self):
        return None

    @property
    def crossing_number(self) -> int:
        return int(self.pts / 2)

    @classmethod
    def from_abstract_state(cls, other: AbstractState) -> 'HighLevelAbstractState':
        obj = cls()
        obj.points = other.points
        obj.edges = other.edges
        obj.faces = other.faces
        return obj
