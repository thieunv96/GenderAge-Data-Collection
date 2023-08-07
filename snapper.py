import cv2
from ultralytics import YOLO
from src.utils.yolov8_pose import Person
from src.reid.tracker import Tracker
import numpy as np
from src.filters.pose_overlap import PoseOverlap
from src.streams.vidgear.gears import CamGear
from src.utils.saver import Saver
import time
import datetime


class Snapper:
    def __init__(self, configs):
        self.model = YOLO('yolov8x-pose.pt')
        self.track = Tracker(num_seq=5)
        self.overlap_checker = PoseOverlap()
        self.frame_id = 0
        self.save_id = 0
        self.url = configs['url']
        self.env_id = configs['env_id']
        self.dataset_dir = configs['datset_dir']
        self.is_stream = configs['is_stream']
        self.target_size = configs['target_size']
        self.target_size = configs['target_size']
        self.num_unprocessed_frames = configs['num_unprocessed_frames']
        self.num_unsaved_frames = configs['num_unsaved_frames']
        self.save_front_only = configs['save_front_only']
        self.min_ratio_height = configs['min_ratio_height']
        self.stream = self.get_stream(self.url)
        self.saver = Saver(self.dataset_dir, self.env_id, self.min_ratio_height, self.save_front_only)



    def get_stream(self,  url):
        if self.is_stream:
            stream = CamGear(source=url, stream_mode = True, logging=False).start()
        else:
            stream = cv2.VideoCapture(url)
        return stream

    def read_frame(self):
        frame = None
        if self.is_stream:
            frame = self.stream.read()
        else:
            _, frame = self.stream.read()
        return frame

    def release(self):
        if self.is_stream:
            self.stream.stop()
        else:
            self.stream.release()


    def run(self):
        start_time = time.time()
        while True:
            frame = self.read_frame()
            if frame is not None:
                if self.frame_id % self.num_unprocessed_frames == 0:
                    detections  = self.model(frame, verbose=False, device="0")[0]
                    keypoints = detections.keypoints.xy
                    if keypoints.shape[0] > 0:
                        if keypoints.shape[1] == 17 and keypoints.shape[2] == 2: 
                            poses = keypoints.cpu().detach().numpy().astype(np.int32)
                            _, pids, _ = self.track(poses)
                            persons = [Person(frame, poses[i], pids[i]) for i in range(keypoints.shape[0])]
                            for person in persons:
                                person.IS_OVERLAP = self.overlap_checker(person, persons)
                                # frame = person.draw_human(frame)
                                if self.save_id % self.num_unsaved_frames == 0:
                                    self.saver.save(person, self.frame_id)
                            self.save_id += 1
                    # cv2.imshow("Output Frame", frame)
                    # # Show output window

                    # key = cv2.waitKey(20) & 0xFF
                    # # check for 'q' key-press
                    # if key == ord("q"):
                    #     #if 'q' key-pressed break out
                    #     break
                self.frame_id += 1
            time.sleep(0.003)
            if self.frame_id % 200 == 0:
                end_time = time.time()
                fps = 200 / (end_time - start_time)
                print(f"[INFO][{datetime.datetime.now()}] Processed: {self.frame_id} frames, FPS: {round(fps, 2)}")
                start_time =  time.time()