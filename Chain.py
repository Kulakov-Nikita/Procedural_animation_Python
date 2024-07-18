from Sphere import Sphere
import torch


class Chain:
    def __init__(self, position: torch.tensor, radius: float, length: int, tile_size: int) -> None:
        self.radius = radius
        self.chain_links: tuple[Sphere] = tuple([Sphere(position=position + torch.tensor((0, 1, 0))*i*radius,
                                                        radius=radius, tile_size=tile_size) for i in range(length)])

    def move(self, move_dir: torch.tensor):
        self.chain_links[0].move(move_dir=torch.nn.functional.normalize(move_dir.reshape((1, 3))).reshape((3,)), speed=0.1)
        anchor = self.chain_links[0]
        for link in self.chain_links[1:]:
            move_dir = (anchor.position - link.position)
            move_dir = torch.nn.functional.normalize(move_dir.reshape((1, 3))).reshape((3,))
            link.move(move_dir=move_dir, speed=0.1)
            anchor = link

    def sdf_capsul(self, rays:torch.tensor, a: torch.tensor, b: torch.tensor):
        pa = rays - a.tile
        ba = b.tile - a.tile
        paba = (pa * ba).sum(axis=1)
        baba = (ba * ba).sum(axis=1)
        h = torch.clamp(paba / baba, 0, 1).unsqueeze(1)
        return torch.sqrt(torch.sum(torch.pow(pa - ba * h, 2), dim=1)) - self.chain_links[0].radius_tile

    def sdf(self, rays: torch.tensor) -> torch.tensor:
        min_sdf = self.sdf_capsul(rays, self.chain_links[0], self.chain_links[1])
        a = self.chain_links[1]
        for link in self.chain_links[2:]:
            min_sdf = torch.minimum(min_sdf, self.sdf_capsul(rays, a, link))
            a = link
        return min_sdf
