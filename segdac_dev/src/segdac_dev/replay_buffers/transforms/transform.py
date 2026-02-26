from abc import ABC
from abc import abstractmethod
from segdac.data.mdp import MdpData


class Transform(ABC):
    def __init__(self, device: str):
        self.device = device

    @abstractmethod
    def apply(self, mdp_data: MdpData) -> MdpData:
        pass
