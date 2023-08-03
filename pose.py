import cv2
import numpy as np


class YoloPose:
    def __init__(self, xy, pid=-1) -> None:
        self.PID = pid
        self.NOSE_IDX = 0
        self.LEFT_EYE_IDX = 1
        self.LEFT_EAR_IDX = 3
        self.LEFT_SHOULDER_IDX = 5
        self.LEFT_ELBOW_IDX = 7
        self.LEFT_WRIST_IDX = 9
        self.LEFT_HIP_IDX = 11
        self.LEFT_KNEE_IDX = 13
        self.LEFT_ANKLE_IDX = 15
        self.RIGHT_EYE_IDX = 2
        self.RIGHT_EAR_IDX = 4
        self.RIGHT_SHOULDER_IDX = 6
        self.RIGHT_ELBOW_IDX = 8
        self.RIGHT_WRIST_IDX = 10
        self.RIGHT_HIP_IDX = 12
        self.RIGHT_KNEE_IDX = 14
        self.RIGHT_ANKLE_IDX = 16

        self.NOSE = xy[self.NOSE_IDX]
        self.LEFT_EYE = xy[self.LEFT_EYE_IDX]
        self.LEFT_EAR = xy[self.LEFT_EAR_IDX]
        self.LEFT_SHOULDER = xy[self.LEFT_SHOULDER_IDX]
        self.LEFT_ELBOW = xy[self.LEFT_ELBOW_IDX]
        self.LEFT_WRIST = xy[self.LEFT_WRIST_IDX]
        self.LEFT_HIP = xy[self.LEFT_HIP_IDX]
        self.LEFT_KNEE = xy[self.LEFT_KNEE_IDX]
        self.LEFT_ANKLE = xy[self.LEFT_ANKLE_IDX]
        self.RIGHT_EYE = xy[self.RIGHT_EYE_IDX]
        self.RIGHT_EAR = xy[self.RIGHT_EAR_IDX]
        self.RIGHT_SHOULDER = xy[self.RIGHT_SHOULDER_IDX]
        self.RIGHT_ELBOW = xy[self.RIGHT_ELBOW_IDX]
        self.RIGHT_WRIST = xy[self.RIGHT_WRIST_IDX]
        self.RIGHT_HIP = xy[self.RIGHT_HIP_IDX]
        self.RIGHT_KNEE = xy[self.RIGHT_KNEE_IDX]
        self.RIGHT_ANKLE = xy[self.RIGHT_ANKLE_IDX]


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
        for p1, p2 in lines:
            cv2.line(frame, p1, p2, (0,255,0), 2)
            cv2.circle(frame, p1, 4, (0,0,255), -1)
            cv2.circle(frame, p2, 4, (0,0,255), -1)
        if self.PID != -1:
            image = cv2.putText(frame, str(self.PID), self.NOSE, cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 1, cv2.LINE_AA)
        return frame