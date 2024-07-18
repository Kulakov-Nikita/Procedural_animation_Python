import torch
from Renderer import Renderer
from Sphere import Sphere
import matplotlib.pyplot as plt

resolution: int = 1000
renderer: Renderer = Renderer(resolution=resolution, steps_per_frame=10)

sph1: Sphere = Sphere(position=torch.tensor((3, 4, 10)), radius=1, tile_size=resolution)
sph2: Sphere = Sphere(position=torch.tensor((3, 0, 10)), radius=1, tile_size=resolution)

frame = renderer.get_frame(objects=[sph1, sph2])

plt.imshow(frame.reshape(resolution, resolution))
plt.show()
