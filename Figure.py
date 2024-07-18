from abc import ABC, abstractmethod
import torch


class Figure(ABC):
    @abstractmethod
    def __init__(self, position: torch.tensor): pass
    @abstractmethod
    def sdf(self, rays: torch.tensor) -> torch.tensor: pass