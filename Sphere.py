from abc import ABC
import torch
from Figure import Figure


class Sphere(Figure, ABC):
    def __init__(self, position: torch.tensor, radius: float, tile_size: int):
        self.position: torch.tensor = position
        self.radius: float = radius
        self.tile = torch.tile(position, (tile_size**2, 1)).cuda()
        self.radius_tile = torch.ones((tile_size ** 2)).cuda()
        self.move_dir = torch.tensor((0.1, 0, 0), dtype=torch.float32)
        self.move_dir_tile = torch.tile(self.move_dir, (tile_size**2, 1)).cuda()

    def move(self):
        self.tile += self.move_dir_tile

    def sdf(self, rays: torch.tensor) -> torch.tensor:
        return torch.sqrt(torch.sum(torch.pow(rays - self.tile, 2), dim=1)) - self.radius_tile
