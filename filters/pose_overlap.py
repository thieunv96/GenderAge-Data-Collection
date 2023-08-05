import numpy as np

class PoseOverlap:
    def __init__(self, pose_ignore_idxs=[], iou_threshold = 0.) -> None:
        self.IGNORE_IDXs = pose_ignore_idxs
        self.iou_threshold = iou_threshold
    
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
    
    def convert_pose(self, pose):
        if len(self.IGNORE_IDXs) == 0:
            return pose
        else:
            pose_cvt = []
            for i, kp in enumerate(pose):
                if i not in self.IGNORE_IDXs:
                    pose_cvt.append(kp)
            pose_cvt = np.array(pose_cvt)
            return pose_cvt

    def cal_bbx(self, pose):
        x_min = np.min(pose[:,0])
        x_max = np.max(pose[:,0])
        y_min = np.min(pose[:,1])
        y_max = np.max(pose[:,1])
        return (x_min, y_min, x_max, y_max)


    def __call__(self, current_pose, all_poses):
        current_pose = self.convert_pose(current_pose)
        all_poses = np.array([self.convert_pose(pose) for pose in all_poses])
        cr_bbx = self.cal_bbx(current_pose)
        is_overlap = False
        for pose in all_poses:
            if not np.array_equal(current_pose, pose):
                other_bbx = self.cal_bbx(pose)
                if self.cal_iou(cr_bbx, other_bbx) > self.iou_threshold:
                    is_overlap = True
                    break
        return is_overlap
