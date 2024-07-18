import cv2
import numpy as np
from Renderer import Renderer
import matplotlib.pyplot as plt


def record_video(path: str, resolution: tuple, fps: int, num_frames: int, renderer: Renderer, objects: tuple, chain) -> None:
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), fps, resolution)
    for i in range(num_frames):
        chain.move()
        frame = renderer.get_frame(objects=objects).reshape((1000, 1000))*255
        frame = np.stack((frame,)*3, axis=2).astype('uint8')
        writer.write(frame.astype('uint8'))
        print(i)
    writer.release()
