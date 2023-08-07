import sys
sys.path.append('src/reid')
import numpy as np
from pytracker.pytracker import DistanceTracker 
import time
from queue import Queue

class Tracker(object):
    """An interface of tracker object for calling from outside
    Args:
        num_seq (int, optional): number of concatenating frames used for action recognition. Defaults to 9.
        image_size (tuple, optional): size of the processing image (width, height in order). Defaults to (1920,1080).
    Attributes:
        configs (dict): configurable parameters of tracker object.
        joint_shape (tuple): shape of the input joint.
        root_tracker (object): tracker object.
        time_queue (Queue): time processing of each frame.
    """
    def __init__(self, num_seq=5, image_size=(1280,720), **kwargs):
        self.configs = self.__get_from_ucfg(num_seq, image_size)
        self.configs['num_seq'] = num_seq
        self.joint_shape = self.configs.get('joint_shape', (17, 2))
        self.root_tracker = DistanceTracker(**self.configs, **kwargs)
        self.time_queue = Queue(maxsize=self.configs['num_seq'])


    def __get_from_ucfg(self, num_seq, image_size):
        """Initialize parameters for an tracker object
        Args:
            num_seq (int): number of concatenating frames used for action recognition.
            image_size (tuple): size of the processing image (width, height in order).
        Returns:
            A dict, containing tracking parameters
        """
        joint_num = 17
        num_seq = num_seq
        track_keypoints = [5,6,11,12]
        track_mode = 'pytracker'
        posefix_mode = 'baseline'
        track_params = dict(num_seq = num_seq,
                            use_iou = False,
                            track_keypoints = track_keypoints,
                            joint_shape = (joint_num, 2),
                            image_size = image_size,
                            dup_offset = num_seq,
                            min_joints = 5,
                            check_neck = False,
                            track_mode = track_mode,
                            posefix_mode = posefix_mode)
        return track_params

    def __getattr__(self, item):
        """Get attribute of the tracker object
        Args:
            item (str): name of the attribute
        Return:
            A value of the attribute
        """
        try:
            return self.root_tracker.__getattribute__(str(item))
        except AttributeError:
            return None 
    
    def __call__(self, poses=None, frame_time=time.time(), frameid=None):
        """An interface for calling the track function
        Args:
            poses (list, optional): list detected poses, containing (x, y, confidence score). Defaults to None.
            frame_time (float, optional): a processing time of the current frame. Defaults to time.time().
            frameid (int, optional): an id of the current frame. Defaults to None.
        Returns:
            An sequence latest poses from previous frames of the input poses respectively
            A list of the pid corresponding with the list input poses
            A list of the processing time of latest frame
        """
        return self.track(poses, frame_time, frameid)

    def track(self, poses, frame_time, frameid):
        """Calling the track function
        Args:
            poses (list, optional): list detected poses, containing (x, y, confidence score). Defaults to None.
            frame_time (float, optional): a processing time of the current frame. Defaults to time.time().
            frameid (int, optional): an id of the current frame. Defaults to None.
        Returns:
            An sequence latest poses from previous frames of the input poses respectively
            A list of the pid corresponding with the list input poses
            A list of the processing time of latest frame
        """
        if poses is None:
            poses = np.zeros((0, *self.joint_shape))
        if frameid is None:
            frameid = -1
        if self.time_queue.empty():
            for _ in range(self.time_queue.maxsize):
                self.time_queue.put(frame_time)
        self.time_queue.get()
        self.time_queue.put(frame_time)
        # Get track
        self.root_tracker(poses, [], frameid)

        # Pose tracks
        pose_tracks, identities = self.root_tracker.tracks 
        nonzero = np.sum(pose_tracks[:, -1, :, 0], axis=1) > 0
        pose_tracks = pose_tracks[nonzero]
        identities  = identities[nonzero]
        if pose_tracks is None:
            pose_tracks = np.zeros((0, self.configs.get('seq_num', self.configs['num_seq']),
                                    *self.configs.get('joint_shape', (17, 2))))
        return pose_tracks, identities, list(self.time_queue.queue)
     


if __name__ == "__main__":
    import time
    from tqdm import tqdm 
    num = 100
    pose_num = 10
    tracker = Tracker()
    poses = np.random.random((pose_num, 17, 2)) * 255
    for _ in tqdm(range(num)):
        pose_tracks, identities, _ = tracker(poses)
