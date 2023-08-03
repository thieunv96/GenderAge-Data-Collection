from scripts.vidgear.gears import CamGear
import cv2
from ultralytics import YOLO
from pose import YoloPose
from tracking.tracker import Tracker
import numpy as np

stream = CamGear(source='https://www.youtube.com/watch?v=gFRtAAmiFbE', stream_mode = True, logging=False) # YouTube Video URL as input
stream = stream.start()
model = YOLO('yolov8x-pose-p6.pt')

track = Tracker()
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
    if c % 1 == 0:
        c = 0
        # frame = cv2.resize(frame, (1280, 720))
        detections  = model(frame, verbose=False)[0]
        if detections.keypoints.xy.shape[0] > 0:
            if detections.keypoints.xy.shape[1] == 17 and detections.keypoints.xy.shape[2] == 2: 
                yolo_poses = detections.keypoints.xy.cpu().detach().numpy().astype(np.int32)
                print(yolo_poses.shape)
                _, pids, _ = track(yolo_poses)
                for kp in range(detections.keypoints.xy.shape[0]):
                    yl = YoloPose(yolo_poses[kp], pids[kp])
                    frame = yl.draw_human(frame)
            
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