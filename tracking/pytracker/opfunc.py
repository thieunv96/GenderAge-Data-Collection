import numpy as np 
import math

def get_bb(single_pose):
    """ Get a bounding box of the input pose
    Args:
        single_pose (list): A single input pose (x, y, confidence score - for all keypoints)
    Returns:
        A bounding box in format [x-topleft, y-topleft, x-bottomright, y-bottomright]
    """
    x_pose = single_pose[:, 0]
    y_pose = single_pose[:, 1]
    x1 = x_pose[x_pose > 0].min()
    x2 = x_pose[x_pose > 0].max()
    y1 = y_pose[y_pose > 0].min()
    y2 = y_pose[y_pose > 0].max()
    return [x1, y1, x2, y2]

def convert_bbox_to_z(bbox):
    """Take a bounding box in the form [x1, y1, x2, y2] 
        And return z in the form [x, y, s, r].
    Where:
        (x, y) - center of the box. s - scale/area. r - the aspect ratio.
    Args:
        bbox (list): a bounding box in format [x-topleft, y-topleft, x-bottomright, y-bottomright]
    Returns:
        A tracking box in the form [x-center, y-center, area, ratio]
    """
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = bbox[0] + w / 2.0
    y = bbox[1] + h / 2.0
    s = w * h
    r = w / float(h + 1e-6)
    return np.array([x, y, s, r]).reshape((4, 1))

def convert_z_to_bbox(z):
    """Take a bounding in the center form [x, y, s, r]
        And return it in the form [x1, y1, x2, y2]
        Where:
            (x1, y1) - top left. (x2, y2) - bottom right
    Args:
        bbox (list): a tracking box in the form [x-center, y-center, area, ratio]
    Returns:
        A bounding box in the form [x-topleft, y-topleft, x-bottomright, y-bottomright]
    """
    w = np.sqrt(z[2] * z[3])
    h = z[2] / w
    return np.array([z[0] - w/2., z[1] - h/2., z[0] + w/2., z[1] + h/2.]).reshape((1, 4))

def speed_direction(bbox1, bbox2):
    """Speed and direction vector from box1 to box2
    Args:
        bbox1 (list): a bounding box in format [x-topleft, y-topleft, x-bottomright, y-bottomright]
        bbox2 (list): a bounding box in format [x-topleft, y-topleft, x-bottomright, y-bottomright]
    Returns:
        A vector moving direction in format (y, x)
    """
    cx1, cy1 = (bbox1[0] + bbox1[2]) / 2.0, (bbox1[1] + bbox1[3]) / 2.0
    cx2, cy2 = (bbox2[0] + bbox2[2]) / 2.0, (bbox2[1] + bbox2[3]) / 2.0
    speed = np.array([cy2 - cy1, cx2 - cx1])
    norm  = np.sqrt((cy2 - cy1)**2 + (cx2 - cx1)**2) + 1e-6
    return speed / norm 

def get_k_previous_obsers(obsers, current_age, k):
    """ Get the k-th previous observations from current frame.
    Args:
        obsers (dict): A dictionary. Key is an frame_id, and value is a pose at this frame.
        current_age (int): A current frame_id
        k (int): The order of the frame, whose pose want to get.
    Returns:
        the k-th previous observation from current.
        Otherwise, the latest observation.
    """
    if len(obsers) == 0:
        return [-1, -1, -1, -1]
    
    for i in range(0, k):
        dt = k - i
        if (current_age - dt) in obsers:
            return obsers[current_age - dt]
    
    max_age = max(obsers.keys())
    return obsers[max_age]

def get_iou_matrix(dets, tracks):
    """ Calculate IoU between an detection and an tracker.
    Args:
        dets (list): A list of detection, each element in format [x-topleft, y-topleft, x-bottomright, y-bottomright]
        tracks (list): A list of trackers, each element in format [x-topleft, y-topleft, x-bottomright, y-bottomright]
    Returns:
        An IoU matrix (NxM) between each element in a detection and tracker list.
    """
    bboxes1 = np.expand_dims(dets, 1)
    bboxes2 = np.expand_dims(tracks, 0)

    xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
    yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
    yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    sa = w * h
    s1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
    s2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1])
    return sa / (s1 + s2 - sa)

def get_area_matrix(bbx1, bbx2):
    h1 = abs(bbx1[..., 3] - bbx1[..., 1])
    w2 = abs(bbx2[..., 2] - bbx2[..., 0])
    w1 = abs(bbx1[..., 2] - bbx1[..., 0])
    h2 = abs(bbx2[..., 3] - bbx2[..., 1])
    
    S1 = w1 * h1
    S2 = w2 * h2
    S1 = np.expand_dims(S1, 1)
    S2 = np.expand_dims(S2, 0)
    
    return np.sqrt((S1 + S2) / 2), np.sqrt(np.fmax(S1, S2) / np.fmin(S1, S2))

def get_distance_matrix_new(dets, tracks, weight):
    bboxes1 = np.expand_dims(dets, 1)
    bboxes2 = np.expand_dims(tracks, 0)

    cx1 = (bboxes1[..., 0] + bboxes1[..., 2]) / 2.0 
    cy1 = (bboxes1[..., 1] + bboxes1[..., 3]) / 2.0
    cx2 = (bboxes2[..., 0] + bboxes2[..., 2]) / 2.0
    cy2 = (bboxes2[..., 1] + bboxes2[..., 3]) / 2.0

    dis = np.sqrt((cx2 - cx1)**2 + (cy2 - cy1)**2)
    s,k = get_area_matrix(dets, tracks)
    
    mtx = dis * k * math.sqrt(weight) / (s + 1e-6)
    mtx = mtx / weight
    mtx = np.where(mtx <= 1, mtx, 1)
    return mtx


def get_distance_matrix(dets, tracks, max_norm=420):
    """ Calculate and normalize the 2D-distance between an detection and an tracker
    Args:
        dets (list): A list of detection, each element in format [x-topleft, y-topleft, x-bottomright, y-bottomright]
        tracks (list): A list of trackers, each element in format [x-topleft, y-topleft, x-bottomright, y-bottomright]
    Returns:
        An distance matrix (NxM) between each element in a detection and tracker list.
    """
    bboxes1 = np.expand_dims(dets, 1)
    bboxes2 = np.expand_dims(tracks, 0)

    cx1 = (bboxes1[..., 0] + bboxes1[..., 2]) / 2.0 
    cy1 = (bboxes1[..., 1] + bboxes1[..., 3]) / 2.0
    cx2 = (bboxes2[..., 0] + bboxes2[..., 2]) / 2.0
    cy2 = (bboxes2[..., 1] + bboxes2[..., 3]) / 2.0

    dis = np.sqrt((cx2 - cx1)**2 + (cy2 - cy1)**2) + 1e-6
    
    # Normalize [0-1]
    for r in range(0, dis.shape[0]):
        for c in range(0, dis.shape[1]):
            if (dis[r, c] > max_norm):
                dis[r, c] = 0.0
            else:
                dis[r, c] = (max_norm - dis[r, c]) / max_norm
    return dis
