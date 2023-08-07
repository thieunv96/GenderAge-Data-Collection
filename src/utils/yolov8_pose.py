import cv2
import numpy as np
import os
import datetime


class Person:
    NOSE_IDX = 0
    LEFT_EYE_IDX = 1
    LEFT_EAR_IDX = 3
    LEFT_SHOULDER_IDX = 5
    LEFT_ELBOW_IDX = 7
    LEFT_WRIST_IDX = 9
    LEFT_HIP_IDX = 11
    LEFT_KNEE_IDX = 13
    LEFT_ANKLE_IDX = 15
    RIGHT_EYE_IDX = 2
    RIGHT_EAR_IDX = 4
    RIGHT_SHOULDER_IDX = 6
    RIGHT_ELBOW_IDX = 8
    RIGHT_WRIST_IDX = 10
    RIGHT_HIP_IDX = 12
    RIGHT_KNEE_IDX = 14
    RIGHT_ANKLE_IDX = 16
    def __init__(self, org_frame, coor, pid=-1) -> None:
        self.ORG_IMG_H = org_frame.shape[0]
        self.PID = pid
        self.COOR = coor
        self.NOSE = self.COOR[Person.NOSE_IDX]
        self.LEFT_EYE = self.COOR[Person.LEFT_EYE_IDX]
        self.LEFT_EAR = self.COOR[Person.LEFT_EAR_IDX]
        self.LEFT_SHOULDER = self.COOR[Person.LEFT_SHOULDER_IDX]
        self.LEFT_ELBOW = self.COOR[Person.LEFT_ELBOW_IDX]
        self.LEFT_WRIST = self.COOR[Person.LEFT_WRIST_IDX]
        self.LEFT_HIP = self.COOR[Person.LEFT_HIP_IDX]
        self.LEFT_KNEE = self.COOR[Person.LEFT_KNEE_IDX]
        self.LEFT_ANKLE = self.COOR[Person.LEFT_ANKLE_IDX]
        self.RIGHT_EYE = self.COOR[Person.RIGHT_EYE_IDX]
        self.RIGHT_EAR = self.COOR[Person.RIGHT_EAR_IDX]
        self.RIGHT_SHOULDER = self.COOR[Person.RIGHT_SHOULDER_IDX]
        self.RIGHT_ELBOW = self.COOR[Person.RIGHT_ELBOW_IDX]
        self.RIGHT_WRIST = self.COOR[Person.RIGHT_WRIST_IDX]
        self.RIGHT_HIP = self.COOR[Person.RIGHT_HIP_IDX]
        self.RIGHT_KNEE = self.COOR[Person.RIGHT_KNEE_IDX]
        self.RIGHT_ANKLE = self.COOR[Person.RIGHT_ANKLE_IDX]
        self.FIT_BOX = self.cal_fit_bbx(self.COOR)
        self.EXTEND_BOX = self.cal_extend_bbx(self.FIT_BOX, org_frame)
        self.IN_EDGE = False
        self.PKP = self.COOR.copy()
        self.PKP[:,0] -= self.EXTEND_BOX[0]
        self.PKP[:,1] -= self.EXTEND_BOX[1]
        self.IMAGE = org_frame[self.EXTEND_BOX[1]:self.EXTEND_BOX[3], self.EXTEND_BOX[0]:self.EXTEND_BOX[2]].copy()
        self.IS_OVERLAP = False

    def cal_extend_bbx(self, fit_box, org_frame):
        ext = 0.2
        x_min, y_min, x_max, y_max = fit_box
        h, w = org_frame.shape[:2]
        dx = x_max - x_min
        dy = y_max - y_min
        new_x_min = max(0, x_min - int(ext * dx))
        new_y_min = max(0, y_min - int(ext * dy))
        new_x_max = min(w-1, x_max + int(ext * dx))
        new_y_max = min(h-1, y_max + int(ext * dy))
        if new_x_min == 0 or new_y_min == 0 or new_x_max == w-1 or new_y_max == h-1:
            self.IN_EDGE = True
        return (new_x_min, new_y_min, new_x_max, new_y_max)


    def cal_fit_bbx(self, pose):
        x_min = np.min(pose[:,0])
        x_max = np.max(pose[:,0])
        y_min = np.min(pose[:,1])
        y_max = np.max(pose[:,1])
        return (x_min, y_min, x_max, y_max)

    def draw_human(self, frame):
        lines = [[self.NOSE, self.LEFT_EYE], 
                 [self.NOSE, self.RIGHT_EYE],
                 [self.LEFT_EYE, self.LEFT_EAR],
                 [self.RIGHT_EYE, self.RIGHT_EYE],
                 [self.NOSE, self.LEFT_SHOULDER],
                 [self.LEFT_SHOULDER, self.LEFT_ELBOW],
                 [self.LEFT_ELBOW, self.LEFT_WRIST],
                 [self.LEFT_SHOULDER, self.LEFT_HIP],
                 [self.LEFT_HIP, self.LEFT_KNEE],
                 [self.LEFT_KNEE, self.LEFT_ANKLE],
                 [self.LEFT_SHOULDER, self.RIGHT_SHOULDER],
                 [self.LEFT_HIP, self.RIGHT_HIP],

                 [self.NOSE, self.RIGHT_SHOULDER],
                 [self.RIGHT_SHOULDER, self.RIGHT_ELBOW],
                 [self.RIGHT_ELBOW, self.RIGHT_WRIST],
                 [self.RIGHT_SHOULDER, self.RIGHT_HIP],
                 [self.RIGHT_HIP, self.RIGHT_KNEE],
                 [self.RIGHT_KNEE, self.RIGHT_ANKLE],
                 ]
        
        line_corlor = (0,0,255) if self.IS_OVERLAP else (0,255,255)
        for p1, p2 in lines:
            cv2.line(frame, p1, p2, line_corlor, 1)
            cv2.circle(frame, p1, 3, (0,0,255), -1)
            cv2.circle(frame, p2, 3, (0,0,255), -1)
        cv2.rectangle(frame, self.EXTEND_BOX[:2], self.EXTEND_BOX[2:], line_corlor, 2)
        if self.PID != -1:
            frame = cv2.putText(frame, str(self.PID), (self.NOSE[0]-10, self.NOSE[1]), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,255,0), 1, cv2.LINE_AA)
        return frame