import torch
from Renderer import Renderer
from Sphere import Sphere
from Chain import Chain
from VideoRecorder import record_video
import matplotlib.pyplot as plt

resolution: int = 1000
renderer: Renderer = Renderer(resolution=resolution, steps_per_frame=10, precision=4)

sph1: Sphere = Sphere(position=torch.tensor((3, 4, 10), dtype=torch.float32), radius=1, tile_size=resolution)
# sph2: Sphere = Sphere(position=torch.tensor((3, 0, 10), dtype=torch.float32), radius=1, tile_size=resolution)
# sph3: Sphere = Sphere(position=torch.tensor((0, 0, 10), dtype=torch.float32), radius=1, tile_size=resolution)
# sph4: Sphere = Sphere(position=torch.tensor((-3, 0, 10), dtype=torch.float32), radius=1, tile_size=resolution)

chain: Chain = Chain(position=torch.tensor((0, 0, 20), dtype=torch.float32), radius=1, length=10, tile_size=resolution)

record_video(path='videos/test.avi', resolution=(resolution, resolution), fps=30, num_frames=90,
             renderer=renderer, objects=[chain, sph1], chain=chain)

#frame = renderer.get_frame(objects=[sph1, sph2])

#plt.imshow(frame.reshape(resolution, resolution))
#plt.show()
