from Sphere import Sphere
import torch


class Chain:
    def __init__(self, position: torch.tensor, radius: float, length: int, tile_size: int) -> None:
        self.radius = radius
        self.chain_links: tuple[Sphere] = tuple([Sphere(position=position + torch.tensor((0, 1, 0))*i*radius,
                                                        radius=radius, tile_size=tile_size) for i in range(length)])

    def move(self):
        move_dir = torch.tensor((1, 0, 0), dtype=torch.float32).cuda()
        self.chain_links[0].move(move_dir=move_dir, speed=1)
        anchor = self.chain_links[0]
        for link in self.chain_links[1:]:
            move_dir = (anchor.position - link.position)
            move_dir = torch.nn.functional.normalize(move_dir.reshape((1, 3))).reshape((3,))
            link.move(move_dir=move_dir, speed=1)
            anchor = link

