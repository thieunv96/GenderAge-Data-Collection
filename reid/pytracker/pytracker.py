import os 
import sys 
import time 
import copy
import numpy as np 

from tqdm import tqdm 
from collections import Counter

from .kalman_filter import KalmanBoxTracker
from .association import associate, linear_assignment
from .opfunc import get_k_previous_obsers, get_iou_matrix, get_distance_matrix, get_distance_matrix_new, get_bb

class DistanceTracker(object):
    """Implementation tracking algorithm
    Args:
        num_seq (int, optional): size of the sequence poses in output. Defaults to 20.
        distance_threshold (float, optional): an threshold for distance matching. Defaults to 0.85.
        joint_shape (tuple, optional): shape of the input joint. Defaults to (17, 3).
        use_iou (bool, optional): use iou or distance metric at the 2nd matching module. Defaults to False.
        track_keypoints (list, optional): which keypoint must be visible. Defaults to [0,5,6,11,12].
        image_size (tuple, optional): size of the processing image (height, width). Defaults to (640, 360).
        **kwargs: Arbitrary keyword arguments.
    Attributes:
        name (str): name of the camera, respects to this tracker
        num_seq (int): size of the sequence poses in output
        image_size (tuple): size of the processing image
        size (tuple): shape of the input joint
        track_keypoints (list): indicate which keypoint must be visible
        use_iou (bool): use iou or distance metric at the 2nd matching module
        iou_threshold (float): an threshold for iou matching
        distance_threshold (float): an threshold for distance matching
        distance_ratio (float): the percentage between maximum acceptable distance matching and input size
        inertia (float): an coefficient of using matrix angle different in the final cost matrix
        max_age (int): maximum number frame, where the pid lost pose
        min_hit (int): minimum number frame, where the pid is detected
        delta_frame (int): number frame, which is used to estimate moving vector
        memory_number (int): maximum number previous frames, which will be saved
        appearances_threshold (int): number minimum appearances for visualizing acceptance
        frame_index (int): a newest frame index of system
        miss_counter (dict): count number missing frames of each pid
        pid_counter (dict): count number appearance frames of each pid
        pid (int): the newest pid of system
        trackers (list): list tracked objects (~ pid)
        frame_list (list): list pose data at historic frames
        track_pose (dict): map a pid and pose index
        tracks (tuple): pose sequence from previous frames to current and list of PID at current frame
    """
    def __init__(self, num_seq=5, distance_threshold=0.85, joint_shape=(15, 3), use_iou=False,
                  track_keypoints=[1,2,5,8,11], image_size=(1920, 1080), **kwargs):
        self.name = kwargs.get('camid', 'pytracker')

        self.num_seq = num_seq
        self.image_size = image_size
        self.size = joint_shape
        self.track_keypoints = track_keypoints
        
        self.use_iou = use_iou 
        self.iou_threshold = 0.1
        self.distance_threshold = distance_threshold
        self.distance_ratio = 7. / 6.
        self.inertia = 0.2
        self.max_age = 30
        self.min_hit = 1
        self.delta_frame = 3
        self.memory_number = 35
        self.appearances_threshold = 5
        self.frame_index = -1
        self.miss_counter = Counter()
        self.pid_counter = Counter()
        self.pid = 0
        self.trackers = []   # List track ~ PID
        self.frame_list = [] # List pose data at historic frames
        self.track_pose = {} # Map a pid and pose index
        self.tracks = (None, None) # Return value
    
    def set_ignore_pids(self, ignore_pid):
        """Add the pid to the ignore list, don't track
        Args:
            ignore_pid (int): an PID, which will be ignored
        """
        self.ignore_pids.add(ignore_pid)
      
    def _ignore_fresh_pids(self):
        """Ignore visualize the PID, which is new, just appears"""
        for idx, pid in enumerate(self.tracks[1]):
            if self.pid_counter[pid] <= self.appearances_threshold:
                self.tracks[1][idx] = -1 # Ignore, don't visualize
        
    def __call__(self, poses=None, list_pose_idx= None,frame_index=-1):
        """An interface for calling the track function
        Args:
            poses (list, optional): a list pose detection at the current frame. Defaults to None.
            list_pose_idx (list, optional): a list index of the occluded pose. Defaults to None. (Not use)
            frame_index (int, optional): a current frame index. Defaults to -1.
        Returns:
            List sequence poses from previous frame to current of each detected pose
            List PID for all detected poses
            List PID for all occluded poses (Not use)
        """
        if poses is None:
            poses = np.zeros((0, *self.size))
        return self._track(poses, list_pose_idx, frame_index)
    
    def _track(self, poses, list_pose_idx, frame_index):
        """Assign PID for each detected pose at the current frame
        Args:
            poses (list): a list pose detection at the current frame
            list_pose_idx (list): a list index of the occlusion pose
            frame_index (int): a current frame index
        Returns:
            List sequence poses from previous frame to current of each detected pose
            List PID for all detected poses
            List PID for all occlusion poses
        """
        self.frame_index = frame_index

        # 1. Get bounding boxes
        valid_index, pose_boxes = [], []
        for i in range(0, poses.shape[0]):
            valid = 0
            this_pose = poses[i, self.track_keypoints]
            for point in this_pose:
                if np.sum(point) > 0.001: 
                    valid += 1
            
            if (valid >= 4):
                valid_index.append(i)
                pose_boxes.append(get_bb(poses[i]))
        pose_data = poses[valid_index]
        pose_boxes = np.array(pose_boxes)

        # OC_SORT combination
        self.track_pose = {}
        result = []
        list_pid_occlusion = []

        if (pose_boxes.shape[0] > 0):
            # 2. TODO: Split detections into high and low score subset
            # 3. Predict next position for all track
            trks, velocities, last_boxes, k_observations = [], [], [], []
            trks_track = {} # map a index in trks and self.trackers
            for i in range(0, len(self.trackers)):
                pos = self.trackers[i].predict()[0]
                if (sum(pos) >= 0):
                    trks.append(pos)
                    trks_track[len(trks) - 1] = i 
                    
                    if self.trackers[i].velocity is not None:
                        velocities.append(self.trackers[i].velocity)
                    else:
                        velocities.append([0, 0])
                    
                    last_boxes.append(self.trackers[i].last_observations)
                    k_observations.append(get_k_previous_obsers(self.trackers[i].observations,
                                                                self.trackers[i].age, self.delta_frame))
            
            # 4. First association for high score subset with predictions KF
            trks, velocities, last_boxes, k_observations = np.array(trks), np.array(velocities), np.array(last_boxes), np.array(k_observations)
            matched, unmatched_id_dets, unmatched_id_trks = associate(pose_boxes,
                    trks, self.iou_threshold, velocities, k_observations, self.inertia)
            for m in matched:
                tracker_id = trks_track[m[1]]
                self.trackers[tracker_id].update(pose_boxes[m[0], :])
                self.track_pose[self.trackers[tracker_id].pid] = m[0]
                if m[0] in list_pose_idx:
                    list_pid_occlusion.append(self.trackers[tracker_id].pid)

                self.pid_counter[self.trackers[tracker_id].pid] += 1
            
            # 5. TODO: Second association for low score subset
            # 6. OCR - recovery
            if (unmatched_id_dets.shape[0] > 0 and unmatched_id_trks.shape[0] > 0):
                left_dets = pose_boxes[unmatched_id_dets]
                left_trks = last_boxes[unmatched_id_trks]

                if self.use_iou:
                    metric_left = get_iou_matrix(left_dets, left_trks)
                    threshold_value = self.iou_threshold
                else:
                    metric_left = get_distance_matrix(left_dets, left_trks, max_norm=min(self.image_size)*self.distance_ratio)
                    # metric_left = get_distance_matrix_new(left_dets, left_trks, weight=min(self.image_size)/self.distance_ratio)
                    threshold_value = self.distance_threshold
                
                if metric_left.max() > threshold_value:
                    matched_indices = linear_assignment(metric_left * (-1))
                    
                    to_remove_dets, to_remove_trks = [], []
                    for m in matched_indices:
                        det_id, trk_id = unmatched_id_dets[m[0]], unmatched_id_trks[m[1]]
                        
                        if (metric_left[m[0], m[1]] < threshold_value):
                            continue 
                        
                        tracker_id = trks_track[trk_id]
                        self.trackers[tracker_id].update(pose_boxes[det_id, :])
                        self.track_pose[self.trackers[tracker_id].pid] = det_id
                        if det_id in list_pose_idx:
                            list_pid_occlusion.append(self.trackers[tracker_id].pid)
                            
                        self.pid_counter[self.trackers[tracker_id].pid] += 1
                        to_remove_dets.append(det_id)
                        to_remove_trks.append(trk_id)
                    
                    unmatched_id_dets = np.setdiff1d(unmatched_id_dets, np.array(to_remove_dets))
                    unmatched_id_trks = np.setdiff1d(unmatched_id_trks, np.array(to_remove_trks))
            
            # 7. OOS - smoothing
            for m in unmatched_id_trks:
                tracker_id = trks_track[m]
                self.trackers[tracker_id].update(None)
            
            # 8. Create a new track for new pid
            for d in unmatched_id_dets:
                trk = KalmanBoxTracker(pose_boxes[d], self.delta_frame, self.pid)
                self.pid += 1
                self.trackers.append(trk)
                self.track_pose[trk.pid] = d
                if d in list_pose_idx:
                    list_pid_occlusion.append(trk.pid)
                self.pid_counter[trk.pid] += 1
            
            # 9. Get value for return and remove dead tracklet pid
            remove_list = []
            for t in range(0, len(self.trackers)):
                if (self.trackers[t].time_since_update < 1):
                    pd = self.track_pose[self.trackers[t].pid]
                    result.append([self.trackers[t].pid, pose_data[pd]])
                
                if (self.trackers[t].time_since_update > self.max_age):
                    remove_list.append(t)
            
            for t in range(len(remove_list) - 1, -1, -1):
                self.trackers.pop(remove_list[t])

        # Upgrade the result store
        self.frame_list.append(result)
        if (len(self.frame_list) > self.memory_number):
            self.frame_list.pop(0)

        # Get sequence poses for the next module
        self.tracks = self.get_sequence()
        
        # Don't show new pids
        # self._ignore_fresh_pids()
        return self.tracks, list_pid_occlusion
    
    
    def get_sequence(self):
        """Concatenate poses from previous frames and current frame for each PID
        Returns:
            List sequence poses from previous frame to current of each detected pose
            List PID of each detected pose
        """
        current_data = self.frame_list[-1]
        pids = [0] * len(current_data)
        for p in current_data:
            pose_index = self.track_pose[p[0]]
            pids[pose_index] = p[0]
        
        sequences = np.zeros((len(pids), self.num_seq, *self.size)).astype(float)
        i = len(self.frame_list) - 1
        while (i >= 0):
            sequence_id = self.num_seq - (len(self.frame_list) - i)
            if (sequence_id < 0):
                break

            current_data = self.frame_list[i]
            for idx in range(0, len(pids)):
                for p in current_data:
                    if (p[0] == pids[idx]):
                        sequences[idx, sequence_id, ::] = p[1]
                        break 
            i -= 1

        self.tracks = (sequences.copy(), np.array(pids).astype(int).copy())
        return self.tracks
    