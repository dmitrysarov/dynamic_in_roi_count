from utils import count_frames, Detections
import numpy as np

detections = Detections(frame_id=np.array([0, 0, 0, 1, 1, 2, 2, 2, 5, 5, 5, 6]),
                        cls=np.array(["mamat", "mamat", "babyt", "mamat", "babyt", "babyt", "mamat", "mamat", "babyt", "mamat", "babyt", "babyt"]),
                        track_id=np.array([0, 1, 2, 0, 2, 2, 3, 0, 0, 1, 1, 4]),
                        x=np.array([0]*12),
                        y=np.array([0]*12),
                        width=np.array([1]*12),
                        height=np.array([1]*12),)

print(count_frames(detections, dynamic_roi_class="mamat"))