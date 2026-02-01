from abc import ABC, abstractmethod
from typing import List, Any
# import matplotlib.pyplot as plt
# import numpy as np
# from tqdm import tqdm


class Schedule(ABC):
    def __init__(self, string: str, enable: bool = True, default_value=0):
        self.name = string
        self.enable = enable
        self.default_value = default_value

    def value(self, step):
        if self.enable:
            return self._value(step)
        else:
            return self.default_value

    @abstractmethod
    def _value(self, step):
        """
        Value of the schedule for a given timestep

        :param step: (int) the timestep
        :return: (float) the output value for the given timestep
        """
        raise NotImplementedError

    # def plot(self, env_steps: np.ndarray = np.arange(1e9)):
    #     title = f'Epsilon Schedule: {self.name}'
    #     values = []
    #     for step in tqdm(env_steps, total=len(env_steps), desc=f'Plotting {title}'):
    #         values.append(self.value(step))
    #     plt.plot(env_steps, values)
    #     plt.title(title)
    #     plt.xlabel("Env Steps")
    #     plt.ylabel("Epsilon")
    #     plt.show()

    def __call__(self, step):
        return self.value(step=step)


class ConstantSchedule(Schedule):
    """
    Value remains constant over time.

    :param value: (float) Constant value of the schedule
    :param string: (str) Name of the schedule
    """

    def __init__(self, value: float, string: str, enable: bool = True, default_value=0):
        super().__init__(string=string, enable=enable, default_value=default_value)
        self._value = value

    def _value(self, step):
        return self._value


def linear_interpolation(left, right, alpha):
    """
    Linear interpolation between `left` and `right`.

    :param left: (float) left boundary
    :param right: (float) right boundary
    :param alpha: (float) coeff in [0, 1]
    :return: (float)
    """

    return left + alpha * (right - left)


def step_interpolation(left, right, alpha):
    return left if alpha < 1 else right


def get_interpolation_by_name(name: str):
    if name == "linear":
        return linear_interpolation
    elif name == "step":
        return step_interpolation
    else:
        raise NotImplementedError


class PiecewiseSchedule(Schedule):
    """
    Piecewise schedule.

    :param endpoints: ([(int, int)])
        list of pairs `(time, value)` meaning that schedule should output
        `value` when `t==time`. All the values for time must be sorted in
        an increasing order. When t is between two times, e.g. `(time_a, value_a)`
        and `(time_b, value_b)`, such that `time_a <= t < time_b` then value outputs
        `interpolation(value_a, value_b, alpha)` where alpha is a fraction of
        time passed between `time_a` and `time_b` for time `t`.
    :param interpolation: (lambda (float, float, float): float)
        a function that takes value to the left and to the right of t according
        to the `endpoints`. Alpha is the fraction of distance from left endpoint to
        right endpoint that t has covered. See linear_interpolation for example.
    :param outside_value: (float)
        if the value is requested outside of all the intervals specified in
        `endpoints` this value is returned. If None then AssertionError is
        raised when outside value is requested.
    :param string: (str) Name of the schedule
    """

    def __init__(self, endpoints: list, string: str, interpolation: str = "linear", outside_value: float = None,
                 enable: bool = True, default_value=0):
        super().__init__(string=string, enable=enable, default_value=default_value)
        idxs = [e[0] for e in endpoints]
        assert idxs == sorted(idxs)
        self._interpolation = get_interpolation_by_name(name=interpolation)
        self._outside_value = outside_value if outside_value is not None else endpoints[-1][1]
        self._endpoints = endpoints

    def _value(self, step):
        for (left_t, left), (right_t, right) in zip(self._endpoints[:-1], self._endpoints[1:]):
            if left_t <= step < right_t:
                alpha = float(step - left_t) / (right_t - left_t)
                return self._interpolation(left, right, alpha)

        # t does not belong to any of the pieces, so doom.
        assert self._outside_value is not None
        return self._outside_value


class LinearSchedule(Schedule):
    """
    Linear interpolation between initial_p and final_p over
    schedule_timesteps. After this many timesteps pass final_p is
    returned.

    :param schedule_timesteps: (int) Number of timesteps for which to linearly anneal initial_p to final_p
    :param initial_p: (float) initial output value
    :param final_p: (float) final output value
    :param string: (str) Name of the schedule
    """

    def __init__(self, string: str, schedule_timesteps: int, final_p: float, initial_p: float = 1.0,
                 enable: bool = True, default_value=0):
        super().__init__(string=string, enable=enable, default_value=default_value)
        self.schedule_timesteps = schedule_timesteps
        self.final_p = final_p
        self.initial_p = initial_p

    def _value(self, step):
        fraction = min(float(step) / self.schedule_timesteps, 1.0)
        v = linear_interpolation(left=self.initial_p, right=self.final_p, alpha=fraction)
        return v


# if __name__ == '__main__':
#     endpoints = [(0, 1.0), (5e3, 1.0), (1e5, 0.0), (1e9, 0.0)]
#     string = f'PieceWise {", ".join([f"{t}:{v}" for t, v in endpoints])}'
#     schedule = PiecewiseSchedule(endpoints=endpoints, string=string, interpolation="linear", outside_value=0.0)
#     schedule.plot(env_steps=np.arange(2e5))
