import numpy as np
import bottleneck as bn
from .opfunc import get_iou_matrix

def estimate_direction_matrix(dets, tracks):
    """Estimate the direction matrix if match an detection with another tracker obj
    Args:
        dets (list): List bounding boxes of detections
        tracks (list): List bounding boxes of tracked objects.
    Returns:
        2 matrixes direction (NxM) by y-axis and x-axis
    """
    tracks = tracks[..., np.newaxis]
    dx1, dy1 = (dets[:, 0] + dets[:, 2]) / 2.0, (dets[:, 1] + dets[:, 3]) / 2.0
    tx1, ty1 = (tracks[:, 0] + tracks[:, 2]) / 2.0, (tracks[:, 1] + tracks[:, 3]) / 2.0
    dx = dx1 - tx1
    dy = dy1 - ty1
    norm = np.sqrt(dx**2 + dy**2) + 1e-6
    return dy / norm, dx / norm # Size: number tracks x number detections

def linear_assignment(cost_matrix):
    """Implementation linear assignment. Find pairs, which cost value is smallest.
    Args:
        cost_matrix (list): Cost matrix between detections and tracked objects.
    Returns:
        List pairs, which cost value is smallest
    """
    id_rows, id_cols = [], []
    cost = cost_matrix.copy()
    for index in range(0, min(cost.shape)):
        minval = bn.nanmin(cost)
        r_ = np.where(cost == minval)
        if len(r_[0]) < 1:
            break
        row, col = r_[0][0], r_[1][0]
        id_rows.append(row)
        id_cols.append(col)
        cost[row, :] = 9999
        cost[:, col] = 9999
    return np.array(list(zip(id_rows, id_cols)))

def associate(dets, tracks, iou_threshold, velocities, prev_obs, vdc_weight):
    """Associate detection with tracked obj base on IoU value.
    Args:
        dets (list): a list bounding boxes of detections
        tracks (list): a list bounding boxes of tracked objects
        iou_threshold (float): a threshold of iou value to verify the assignment result
        velocities (list): a list vectors direction (y-axis, x-axis) of tracked objects
        prev_obs (list): a list previous observations of tracked objects 
        vdc_weight (float): an coefficient of matrix angle difference in the final cost matrix
    Returns:
        List ID in detections and in tracked list, which match to each other.
        List ID in detections, which is free.
        List ID in tracked list, which is free
    """
    if len(tracks) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(dets)), np.empty((0, 5), dtype=int)
    
    tracks[tracks < 0] = 0
    dets[dets < 0] = 0 
    
    # Get angle difference matrix
    vy, vx = estimate_direction_matrix(dets, prev_obs)
    inertia_y, inertia_x = velocities[:, 0], velocities[:, 1]
    inertia_y = np.repeat(inertia_y[:, np.newaxis], vy.shape[1], axis=1)
    inertia_x = np.repeat(inertia_x[:, np.newaxis], vx.shape[1], axis=1)
    angle_diff = inertia_x * vx + inertia_y * vy            # Vector multiplication
    angle_diff = np.clip(angle_diff, a_min=-1, a_max=1)
    angle_diff = np.arccos(angle_diff)
    angle_diff = (np.pi / 2.0 - np.abs(angle_diff)) / np.pi # Normalization [-0.5, 0.5]
    angle_diff = (angle_diff * vdc_weight).T * 0.9          # Score value ~ 0.9

    # Get iou matrix
    iou_matrix = get_iou_matrix(dets, tracks)

    # Linear assignment
    if min(iou_matrix.shape) > 0:
        cost = (iou_matrix + angle_diff) * (-1.0)
        matched_indices = linear_assignment(cost) # Cost matrix with OCM term
    else:
        matched_indices = np.empty(shape=(0, 2))
    
    unmatched_detections = []
    for d in range(0, len(dets)):
        if (d not in matched_indices[:, 0]):
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t in range(0, len(tracks)):
        if (t not in matched_indices[:, 1]):
            unmatched_trackers.append(t)
    
    # Filter out matched with low IoU
    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)
    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)