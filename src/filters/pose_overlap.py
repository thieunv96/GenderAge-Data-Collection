import numpy as np
from src.utils.yolov8_pose import Person


class PoseOverlap:
    def __init__(self) -> None:
        pass

    def cal_iou(self, boxA, boxB):
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)
        # return the intersection over union value
        return iou
    


    def __call__(self, person:Person, persons:list):
        cr_bbx = person.EXTEND_BOX
        is_overlap = False
        for o_person in persons:
            if not np.array_equal(person.COOR, o_person.COOR):
                other_bbx = o_person.EXTEND_BOX
                if self.cal_iou(cr_bbx, other_bbx) > 0:
                    is_overlap = True
                    break
        return is_overlap