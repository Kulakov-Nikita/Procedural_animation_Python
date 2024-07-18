from abc import ABC
import torch
from Figure import Figure


class Sphere(Figure, ABC):
    def __init__(self, position: torch.tensor, radius: float, tile_size: int):
        self.position: torch.tensor = position.cuda()
        self.radius: float = radius
        self.tile = torch.tile(position, (tile_size**2, 1)).cuda()
        self.tile_size: int = tile_size
        self.radius_tile = torch.ones((tile_size ** 2)).cuda()*radius

    def move(self, move_dir: torch.tensor, speed: float):
        move_dir_tile = torch.tile(move_dir, (self.tile_size ** 2, 1))
        self.tile += move_dir_tile * speed
        self.position += move_dir * speed

    def sdf(self, rays: torch.tensor) -> torch.tensor:
        return torch.maximum(torch.sqrt(torch.sum(torch.pow(rays - self.tile, 2), dim=1)), self.radius_tile) - self.radius_tile
