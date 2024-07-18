import cv2
import torch
import numpy as np
from Renderer import Renderer
import matplotlib.pyplot as plt


def record_video(path: str, resolution: tuple, fps: int, num_frames: int, renderer: Renderer, objects: tuple, chain) -> None:
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), fps, resolution)
    for i in range(num_frames):

        match i//30:
            case 0:
                chain.move(torch.tensor((1, 0, 0), dtype=torch.float32).cuda())
            case 1:
                chain.move(torch.tensor((0, 1, 0), dtype=torch.float32).cuda())
            case 2:
                chain.move(torch.tensor((-1, 0, -0.5), dtype=torch.float32).cuda())

        frame = renderer.get_frame(objects=objects).reshape((1000, 1000))*255
        frame = np.stack((frame,)*3, axis=2).astype('uint8')
        writer.write(frame.astype('uint8'))
        print(i)
    writer.release()
