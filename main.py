from streams.vidgear.gears import CamGear
import cv2
from ultralytics import YOLO
from utils.yolov8_pose import Person
from reid.tracker import Tracker
import numpy as np
from filters.pose_overlap import PoseOverlap
from utils.stream_source import STREAMS

stream = CamGear(source=STREAMS[21], stream_mode = True, logging=False) # YouTube Video URL as input
stream = stream.start()
model = YOLO('yolov8n-pose.pt')
track = Tracker()
overlap_checker = PoseOverlap()
skip = False
c = 0
while True:
    skip = not skip
    
    frame = stream.read()
    # read frames
    if skip:
        continue

    # check if frame is None
    if frame is None:
        #if True break the infinite loop
        break
    c += 1
    if c % 6 == 0:
        c = 0
        frame = cv2.resize(frame, (1280, 720))
        detections  = model(frame, verbose=False)[0]
        if detections.keypoints.xy.shape[0] > 0:
            if detections.keypoints.xy.shape[1] == 17 and detections.keypoints.xy.shape[2] == 2: 
                presons = detections.keypoints.xy.cpu().detach().numpy().astype(np.int32)
                _, pids, _ = track(presons)
                for i in range(detections.keypoints.xy.shape[0]):
                    is_overlap = overlap_checker(presons[i], presons)
                    p = Person(presons[i], pids[i])
                    frame = p.draw_human(frame, is_overlap)
            
        cv2.imshow("Output Frame", frame)
        # Show output window

        key = cv2.waitKey(20) & 0xFF
        # check for 'q' key-press
        if key == ord("q"):
            #if 'q' key-pressed break out
            break

cv2.destroyAllWindows()
# close output window

# safely close video stream.
stream.stop()