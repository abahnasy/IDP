
import numpy as np
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

class WaymoDatasetConfig(object):
    def __init__(self):
        self.num_class = 1
        self.num_heading_bin = 12 # TODO: Figure out what is this?!
        self.num_size_cluster = 1 # same as number of classes

        # self.type2class = {'TYPE_UNKNOWN':0,'TYPE_VEHICLE':1,'TYPE_PEDESTRIAN':2,'TYPE_SIGN':3,'TYPE_CYCLIST':4}
        # used to convert original labels to custom labels according to the num classes used
        self.raw2used_labels = {1:0}
        self.type2class = {'TYPE_VEHICLE': 0} #, 'TYPE_PEDESTRIAN': 1, 'TYPE_CYCLIST': 2}
        self.class2type = {self.type2class[t]:t for t in self.type2class}
        # self.type2onehotclass={'TYPE_VEHICLE':0, 'TYPE_PEDESTRIAN':1, 'TYPE_CYCLIST':2}
        
        # run --compute_median_size to get these values based on the current data you have
        self.type_mean_size = {'TYPE_VEHICLE': np.array([4.628242,2.075742,1.690000])} #,
                        #   'TYPE_PEDESTRIAN': np.array([0.913247,0.846451,1.758479]),
                        #   'TYPE_CYCLIST': np.array([1.846484,0.834043,1.800000])}

        self.mean_size_arr = np.zeros((self.num_size_cluster, 3))
        for i in range(self.num_size_cluster):
            self.mean_size_arr[i,:] = self.type_mean_size[self.class2type[i]]

    def size2class(self, size, type_name):
        ''' Convert 3D box size (l,w,h) to size class and size residual '''
        size_class = self.type2class[type_name]
        size_residual = size - self.type_mean_size[type_name]
        return size_class, size_residual
    
    def class2size(self, pred_cls, residual):
        ''' Inverse function to size2class '''
        mean_size = self.type_mean_size[self.class2type[pred_cls]]
        return mean_size + residual
    
    def angle2class(self, angle):
        ''' Convert continuous angle to discrete class
            [optinal] also small regression number from  
            class center angle to current angle.
           
            angle is from 0-2pi (or -pi~pi), class center at 0, 1*(2pi/N), 2*(2pi/N) ...  (N-1)*(2pi/N)
            return is class of int32 of 0,1,...,N-1 and a number such that
                class*(2pi/N) + number = angle
        '''
        num_class = self.num_heading_bin
        angle = angle%(2*np.pi)
        assert(angle>=0 and angle<=2*np.pi)
        angle_per_class = 2*np.pi/float(num_class)
        shifted_angle = (angle+angle_per_class/2)%(2*np.pi)
        class_id = int(shifted_angle/angle_per_class)
        residual_angle = shifted_angle - (class_id*angle_per_class+angle_per_class/2)
        return class_id, residual_angle
    
    def class2angle(self, pred_cls, residual, to_label_format=True):
        ''' Inverse function to angle2class '''
        num_class = self.num_heading_bin
        angle_per_class = 2*np.pi/float(num_class)
        angle_center = pred_cls * angle_per_class
        angle = angle_center + residual
        if to_label_format and angle>np.pi:
            angle = angle - 2*np.pi
        return angle

    def param2obb(self, center, heading_class, heading_residual, size_class, size_residual):
        heading_angle = self.class2angle(heading_class, heading_residual)
        box_size = self.class2size(int(size_class), size_residual)
        obb = np.zeros((7,))
        obb[0:3] = center
        obb[3:6] = box_size
        obb[6] = heading_angle*-1
        return obb


