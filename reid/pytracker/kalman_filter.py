import math 
import copy
import numpy as np 

from .opfunc import convert_bbox_to_z, convert_z_to_bbox, speed_direction

class KalmanFilter(object):
    """ Implementation of KalmanFilter algorithm. 
    Args:
        dim_x (int): size of the x-dimensions.
        dim_z (int): size of the z-dimensions (input).
        dim_u (int): size of the y-dimensions (not use).
    Attribute:
        dim_x (int): size of the x-dimensions.
        dim_z (int): size of the z-dimensions (input).
        dim_u (int): size of the y-dimensions (not used).
        attr_saved (dict): save the attributes at the latest frame, where obj is detected.
        observed (bool): is this object detected.
    References
        ----------
        .. [1] Dan Simon. "Optimal State Estimation." John Wiley & Sons.
          p. 208-212. (2006)
        .. [2] Roger Labbe. "Kalman and Bayesian Filters in Python"
          https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python
    """
    def __init__(self, dim_x, dim_z, dim_u=0):
        assert(dim_x > 0 and dim_z > 0 and dim_u >= 0)
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.dim_u = dim_u

        self.x = np.zeros((dim_x, 1))        # state
        self.P = np.eye(dim_x)               # uncertainty covariance
        self.Q = np.eye(dim_x)               # process uncertainty
        # self.B = None                      # control transition matrix
        self.F = np.eye(dim_x)               # state transition matrix
        self.H = np.zeros((dim_z, dim_x))    # measurement function
        self.R = np.eye(dim_z)               # measurement uncertainty
        self._alpha_sq = 1.                  # fading memory control
        self.M = np.zeros((dim_x, dim_z))    # process-measurement cross correlation
        self.z = np.array([[None]*self.dim_z]).T

        # gain and residual are computed during the innovation step. We
        # save them so that in case you want to inspect them for various purposes
        self.K  = np.zeros((dim_x, dim_z))   # kalman gain
        self.y  = np.zeros((dim_z, 1))
        self.S  = np.zeros((dim_z, dim_z))   # system uncertainty
        self.SI = np.zeros((dim_z, dim_z))   # inverse system uncertainty

        # identity matrix. Do not alter this.
        self._I = np.eye(dim_x)

        # these will always be a copy of x,P after predict() is called
        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()

        # these will always be a copy of x,P after update() is called
        self.x_post = self.x.copy()             
        self.P_post = self.P.copy()

        # keep all observations 
        self.history_obs = []
        self.inv = np.linalg.inv

        self.attr_saved = None
        self.observed = False
    
    def predict(self):
        """Predict next value of the object"""
        # x = Fx + Bu
        self.x = np.dot(self.F, self.x)

        # P = FPF' + Q
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) * self._alpha_sq + self.Q

        # Save prior
        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()
    
    def freeze(self):
        """Save the parameters before non-observation forward"""
        self.attr_saved = copy.deepcopy(self.__dict__)
    
    def unfreeze(self):
        """Correct attributes of the object during untracked period"""
        if self.attr_saved is not None:
            new_history = copy.deepcopy(self.history_obs)
            self.__dict__ = self.attr_saved
            self.history_obs = self.history_obs[:-1]

            occured = np.array([int(d is None) for d in new_history])
            indices = np.where(occured == 0)[0]
            id1, id2 = indices[-2:]
            box1, box2 = new_history[id1], new_history[id2]
            
            x1, y1, s1, r1 = box1
            w1 = np.sqrt(s1 * r1)
            h1 = np.sqrt(s1 / r1)
            x2, y2, s2, r2 = box2
            w2 = np.sqrt(s2 * r2)
            h2 = np.sqrt(s2 / r2)
            time_gap = id2 - id1 
            dx, dy = (x2 - x1) / time_gap, (y2 - y1) / time_gap
            dw, dh = (w2 - w1) / time_gap, (h2 - h1) / time_gap
            for i in range(0, time_gap):
                # Virtual tracjectory generation is linear motion (constant speed hypothesis)
                x, y = x1 + (i + 1) * dx, y1 + (i + 1) * dy
                w, h = w1 + (i + 1) * dw, h1 + (i + 1) * dh
                s, r = w * h, w / float(h)
                new_box = np.array([x, y, s, r]).reshape((4, 1))
                self.update(new_box)
                if (i != (time_gap - 1)):
                    self.predict()

    def update(self, z):
        """Update current attributes of the object"""
        self.history_obs.append(z)
        if z is None:
            if self.observed:
                self.freeze()
            
            self.observed = False
            self.z = np.array([[None] * self.dim_z]).T
            self.x_post = self.x.copy()
            self.P_post = self.P.copy()
            self.y = np.zeros((self.dim_z, 1))
            return
        
        if not self.observed:
            # Get observations, use online smoothing to re-update parameters
            self.unfreeze()
        self.observed = True 

        # y = z - Hx: error (residual) between measurement and prediction
        self.y = z - np.dot(self.H, self.x)
        # Common subexpression for speed
        PHT = np.dot(self.P, self.H.T)
        # S = HPH' + R: project system uncertainty into measurement space
        self.S = np.dot(self.H, PHT) + self.R
        self.SI = self.inv(self.S)
        # K = PH'inv(S): map system uncertainty into kalman gain
        self.K = np.dot(PHT, self.SI)
        # x = x + Ky: predict new x with residual scaled by the kalman gain
        self.x = self.x + np.dot(self.K, self.y)
        # P = (I - KH)P(I-KH)' + KRK'
        # This is more numerically stable and works for non-optimal K vs equation
        # P = (I - KH)P usually seen in the literature
        I_KH = self._I - np.dot(self.K, self.H)
        self.P = np.dot(np.dot(I_KH, self.P), I_KH.T) + \
                 np.dot(np.dot(self.K, self.R), self.K.T)
        
        # Save measurement and posterior state
        self.z = copy.deepcopy(z)
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()
    
class KalmanBoxTracker(object):
    """This class represents the internal state of individual tracked objects observed as bbox.
    Args:
        bbox (list): a first detection box, in form [x-topleft, y-topleft, x-bottomright, y-bottomright]
        delta_frame (int): a range to get the velocity
        pid (int): a pid of this tracked object
    Attributes:
        dim_x (int): size of the x-dimensions
        dim_z (int): size of the z-dimensions (input)
        kalman_filter (obj): An object represents the KF algorithm
        time_since_update (int): how many frames obj is not detected
        delta_t (int): a range to get the velocity
        pid (int): a pid of this tracked object
        age (int): a frame index
        hits (int): how many frames obj is detected, from the first detections
        hit_streak (int): how many frames obj is detected, from the latest detections
        velocity (list): the speed by y-axis, x-axis
        history (list): list predictions in lost pose frames
        history_observations (list): list detected of the object
        last_observations (list): an latest observation
        observations (dict): a map between frame_id and pose at this frame
    """
    def __init__(self, bbox, delta_frame, pid):
        self.dim_x = 7
        self.dim_z = 4
        
        self.kalman_filter = KalmanFilter(dim_x=self.dim_x, dim_z=self.dim_z)
        self.kalman_filter.F = np.array([[1, 0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0, 1], 
                                         [0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0],
                                         [0, 0, 0, 0, 0, 0, 1]])
        self.kalman_filter.H = np.array([[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0],
                                         [0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]])
        self.kalman_filter.R[2:, 2:] *= 10.
        self.kalman_filter.P[4:, 4:] *= 1000.
        self.kalman_filter.P *= 10.
        self.kalman_filter.Q[-1, -1] *= 0.01
        self.kalman_filter.Q[4:, 4:] *= 0.01
        self.kalman_filter.x[:4] = convert_bbox_to_z(bbox)

        self.time_since_update = 0
        self.delta_t = delta_frame
        self.pid = pid
        self.age = 0
        self.hits = 0
        self.hit_streak = 0

        self.velocity = None
        self.history = []
        self.history_observations = []
        self.last_observations = np.array([-1, -1, -1, -1])
        self.observations = dict()
    
    def update(self, bbox):
        """Update the state vector with observed bbox
        Args:
            bbox (list): the observed box, in form [x-topleft, y-topleft, x-bottomright, y-bottomright]
        """
        if bbox is not None:
            if self.last_observations.sum() >= 0:
                previous_box = None
                for i in range(0, self.delta_t):
                    dt = self.delta_t - i
                    if (self.age - dt) in self.observations:
                        previous_box = self.observations[self.age - dt]
                        break 
                if previous_box is None:
                    previous_box = self.last_observations
                self.velocity = speed_direction(previous_box, bbox)
            
            self.last_observations = bbox 
            self.observations[self.age] = bbox
            self.history_observations.append(bbox)

            self.time_since_update = 0
            self.history = []
            self.hits += 1
            self.hit_streak += 1
            self.kalman_filter.update(convert_bbox_to_z(bbox))
        else:
            self.kalman_filter.update(bbox)
    
    def predict(self):
        """Advances the state vector and return the predicted box"""
        if ((self.kalman_filter.x[6] + self.kalman_filter.x[2]) <= 0):
            self.kalman_filter.x[6] *= 0.0
        
        self.kalman_filter.predict()
        self.age += 1
        if (self.time_since_update > 0):
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_z_to_bbox(self.kalman_filter.x))
        return self.history[-1]