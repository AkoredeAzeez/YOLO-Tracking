import cv2
from cv2.typing import MatLike

class VideoWriter:
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.out: cv2.VideoWriter | None = None
    
    def write(self, img: MatLike):
        if not self.out:
            height, width, _ = img.shape
            self.out = cv2.VideoWriter(self.video_path, self.fourcc, 30, (width, height))
        self.out.write(img)
    
    def cleanup(self):
        if self.out:
            self.out.release()
            self.out = None