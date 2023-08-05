import cv2
import numpy as np

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
    def __init__(self, xy, pid=-1) -> None:
        self.PID = pid
        self.XY = xy
        self.NOSE = xy[Person.NOSE_IDX]
        self.LEFT_EYE = xy[Person.LEFT_EYE_IDX]
        self.LEFT_EAR = xy[Person.LEFT_EAR_IDX]
        self.LEFT_SHOULDER = xy[Person.LEFT_SHOULDER_IDX]
        self.LEFT_ELBOW = xy[Person.LEFT_ELBOW_IDX]
        self.LEFT_WRIST = xy[Person.LEFT_WRIST_IDX]
        self.LEFT_HIP = xy[Person.LEFT_HIP_IDX]
        self.LEFT_KNEE = xy[Person.LEFT_KNEE_IDX]
        self.LEFT_ANKLE = xy[Person.LEFT_ANKLE_IDX]
        self.RIGHT_EYE = xy[Person.RIGHT_EYE_IDX]
        self.RIGHT_EAR = xy[Person.RIGHT_EAR_IDX]
        self.RIGHT_SHOULDER = xy[Person.RIGHT_SHOULDER_IDX]
        self.RIGHT_ELBOW = xy[Person.RIGHT_ELBOW_IDX]
        self.RIGHT_WRIST = xy[Person.RIGHT_WRIST_IDX]
        self.RIGHT_HIP = xy[Person.RIGHT_HIP_IDX]
        self.RIGHT_KNEE = xy[Person.RIGHT_KNEE_IDX]
        self.RIGHT_ANKLE = xy[Person.RIGHT_ANKLE_IDX]


    def draw_human(self, frame, is_overlap=False):
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
            line_corlor = (0,0,255) if is_overlap else (0,255,255)
            cv2.line(frame, p1, p2, line_corlor, 1)
            cv2.circle(frame, p1, 3, (0,0,255), -1)
            cv2.circle(frame, p2, 3, (0,0,255), -1)
        if self.PID != -1:
            frame = cv2.putText(frame, str(self.PID), (self.NOSE[0]-10, self.NOSE[1]), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,255,0), 1, cv2.LINE_AA)
        return frame