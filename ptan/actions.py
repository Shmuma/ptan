import abc
import numpy as np
import typing as tt


class ActionSelector(abc.ABC):
    """
    Abstract class which converts scores to the actions
    """
    @abc.abstractmethod
    def __call__(self, scores: np.ndarray) -> np.ndarray:
        ...


class ArgmaxActionSelector(ActionSelector):
    """
    Selects actions using argmax
    """
    def __call__(self, scores: np.ndarray) -> np.ndarray:
        return np.argmax(scores, axis=1)


class EpsilonGreedyActionSelector(ActionSelector):
    def __init__(self, epsilon: float = 0.05,
                 selector: tt.Optional[ActionSelector] = None):
        self._epsilon = epsilon
        self.selector = selector if selector is not None \
            else ArgmaxActionSelector()

    @property
    def epsilon(self) -> float:
        return self._epsilon

    @epsilon.setter
    def epsilon(self, value: float):
        if value < 0.0 or value > 1.0:
            raise ValueError("Epsilon has to be between 0 and 1")
        self._epsilon = value

    def __call__(self, scores: np.ndarray) -> np.ndarray:
        assert len(scores.shape) == 2
        batch_size, n_actions = scores.shape
        actions = self.selector(scores)
        mask = np.random.random(size=batch_size) < self.epsilon
        rand_actions = np.random.choice(n_actions, sum(mask))
        actions[mask] = rand_actions
        return actions


class ProbabilityActionSelector(ActionSelector):
    """
    Converts probabilities of actions into action by sampling them
    """
    def __call__(self, probs: np.ndarray) -> np.ndarray:
        actions = []
        for prob in probs:
            actions.append(np.random.choice(len(prob), p=prob))
        return np.array(actions)


class EpsilonTracker:
    """
    Updates epsilon according to linear schedule
    """
    def __init__(self, selector: EpsilonGreedyActionSelector,
                 eps_start: tt.Union[int, float],
                 eps_final: tt.Union[int, float],
                 eps_frames: int):
        self.selector = selector
        self.eps_start = eps_start
        self.eps_final = eps_final
        self.eps_frames = eps_frames
        self.frame(0)

    def frame(self, frame: int):
        eps = self.eps_start - frame / self.eps_frames
        self.selector.epsilon = max(self.eps_final, eps)
