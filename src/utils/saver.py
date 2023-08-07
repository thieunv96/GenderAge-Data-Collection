import os
import numpy as np
import cv2
import datetime

class Saver:
    def __init__(self, datset_path, env_idx, min_ratio_height=0.2, front_only=False):
        now = datetime.datetime.now()
        self.grab_time = f"{now.month}{now.day}{now.hour}{now.minute}{now.second}"
        self.env_idx = env_idx
        self.min_ratio_height = min_ratio_height
        self.front_only = front_only
        self.datset_path =  f"{datset_path}/{self.grab_time}_{self.env_idx}"
        os.makedirs(self.datset_path , exist_ok=True)
        self.pose_csv = open(f"{self.datset_path}_poses.csv", "a+")
        self.pose_csv.write("FILE_NAME, POSE")

    def is_can_save(self, person):
        if person.IN_EDGE:
            return False
        h, w = person.IMAGE.shape[:2]
        if person.IMAGE.shape[0] < self.min_ratio_height * person.ORG_IMG_H:
            return False
        if 1.3 * w >= h or w <= 0.2*h:
            return False
        if self.front_only and person.RIGHT_SHOULDER[0] > person.LEFT_SHOULDER[0]:
            return  False
        return True

    def save(self, person, frm_idx):
        if self.is_can_save(person):
            dst_dir = f"{self.datset_path}/P{person.PID}"
            os.makedirs(dst_dir, exist_ok=True)
            fname = f"{dst_dir}/{frm_idx}.jpeg"
            cv2.imwrite(fname, person.IMAGE, (cv2.IMWRITE_JPEG_QUALITY, 100))
            pose_1d = person.PKP.flatten()
            str_pose = str(list(pose_1d))
            self.pose_csv.write(f"\n{fname},\"{str_pose}\"")