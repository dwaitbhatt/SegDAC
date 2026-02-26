from abc import ABC
from abc import abstractmethod
from tensordict import TensorDict


class Transform(ABC):
    @abstractmethod
    def apply(self, data: TensorDict) -> TensorDict:
        pass
