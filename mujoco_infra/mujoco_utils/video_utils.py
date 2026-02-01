import os
from typing import List, Optional
import cv2
import numpy as np
from tqdm import tqdm


class Video:
    """
    A class that represents a Video
    """
    def __init__(self, extension: str = '.mp4', upsample_rate: int = 1, min_frames: int = 1):
        self._frames = []
        self.extension = extension
        self.upsample_rate = upsample_rate
        self.min_frames = min_frames
        self._text = []

    @property
    def frames(self) -> List[np.ndarray]:
        return self._frames

    def add_frames(self, frames: List[np.ndarray], texts: Optional[List[str]] = None):
        self._frames.extend(frames)
        self._text.extend(texts or [None] * len(frames))

    def save(self, output_path: str, filename: Optional[str] = None, force: bool = False):
        if filename is None:
            video_index = 0
            free = True
            while free:
                if os.path.isfile(output_path + str(video_index) + f"_project{self.extension}") and not force:
                    video_index += 1
                else:
                    free = False
            path = output_path + str(video_index) + f"_project{self.extension}"
        else:
            if not filename.endswith(self.extension):
                filename += self.extension
            path = os.path.join(output_path, filename)

        # print(f'Saving video to path {path}')
        os.makedirs(output_path, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        height, width, _ = self._frames[0].shape
        height *= self.upsample_rate  # upsample
        width *= self.upsample_rate  # upsample
        frame_rate = 60
        video_writer = cv2.VideoWriter(path, fourcc, frame_rate, (width, height))

        frames = self._frames
        if len(frames) < self.min_frames:
            # find number of times to repeat each frame
            repeats = (self.min_frames // len(frames)) + 1
            frames = [frame for frame in frames for _ in range(repeats)]

        for i, frame in tqdm(enumerate(frames), total=len(frames), desc="Saving video"):
            upsampled_frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
            frame_bgr = cv2.cvtColor(upsampled_frame, cv2.COLOR_RGB2BGR)
            text = self._text[i]
            if text is not None:
                cv2.putText(frame_bgr, text, (width // 2 - 100, height // 2 - 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            video_writer.write(frame_bgr)

        video_writer.release()
        print(f"Video saved to {path}")
