import torch
from Renderer import Renderer
from Sphere import Sphere
from Chain import Chain
from VideoRecorder import record_video
import matplotlib.pyplot as plt
import sys


def main():
    resolution: int = int(sys.argv[1])
    output_path: str = sys.argv[2]
    fps: int = int(sys.argv[3])
    num_frames: int = int(sys.argv[4])
    objects = []
    chain=None
    for i in range(5, len(sys.argv), 5):
        if sys.argv[i] in ['-s', '--sphere']:
            objects.append(Sphere(position=torch.tensor((int(sys.argv[i+1]),int(sys.argv[i+2]),int(sys.argv[i+3])),dtype=torch.float32),
                                  radius=int(sys.argv[i+4]), tile_size=resolution))
        if sys.argv[i] in ['-c', '--chain']:
            if chain == None:
                chain = Chain(position=torch.tensor((int(sys.argv[i+1]),int(sys.argv[i+2]),int(sys.argv[i+3])),
                                                    dtype=torch.float32), radius=int(sys.argv[i+4]), length=10,
                                                    tile_size=resolution)
                objects.append(chain)
            else:
                print("There can only be one Chain object at a time.")
                return

    record_video(path=output_path, resolution=(resolution, resolution), fps=fps, num_frames=num_frames,
                 renderer=Renderer(resolution), objects=objects, chain=chain)



if __name__ == '__main__':
    main()