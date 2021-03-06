""" Waymo dataset with votes.

Author: Ahmed Bahnasy
Date: 2020

"""
import os
import sys
import numpy as np
import pickle
from torch.utils.data import Dataset
import scipy.io as sio # to load .mat files for depth points
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '..', 'utils'))
from box_util import get_corners_from_labels_array
import pc_util
import waymo_utils
from model_util_waymo import WaymoDatasetConfig

DC = WaymoDatasetConfig() # dataset specific config
MAX_NUM_OBJ = 128 # maximum number of objects allowed per scene

# RAW_LABELS =  {0: 'TYPE_UNKNOWN', 1: 'TYPE_VEHICLE' , 2: 'TYPE_PEDESTRIAN', 3: 'TYPE_SIGN', 4: 'TYPE_CYCLIST'}

class WaymoDetectionVotesDataset(Dataset):
    def __init__(self, split_set='train', num_points=180000,
        use_height=False,
        augment=False,
        verbose:bool = True):
        # self.mapping_labels = {1:0,2:1,4:2} # map dataset labels to our labels to handle discarded classes 
        # self.excluded_labels = [0,3] # exclude unknowns and signs labels
        self.split_set = split_set
        self.type2class = {0: 'TYPE_UNKNOWN', 1: 'TYPE_VEHICLE' , 2: 'TYPE_PEDESTRIAN', 3: 'TYPE_SIGN', 4: 'TYPE_CYCLIST'}
        self.class2type = {self.type2class[t]:t for t in self.type2class}
        self.classes = ['TYPE_VEHICLE'] #, 'TYPE_PEDESTRIAN', 'TYPE_CYCLIST']
        self.data_path = os.path.join(BASE_DIR,
                'dataset') # TODO: rename to votes data path

        # self.raw_data_path = os.path.join(BASE_DIR, 'dataset')
        
        # access segments dictionary list
        # load segments_dict_list dictionary
         
        self.segments_dict_list_path = os.path.join(self.data_path, split_set, 'segments_dict_list')
        if not os.path.exists(self.segments_dict_list_path):
            raise ValueError('segments Dictionary list is not found, make sure to preprocess the data first')
        with open(self.segments_dict_list_path, 'rb') as f:
            self.segments_dict_list = pickle.load(f)
        

        self.num_segments = len(self.segments_dict_list)
        if verbose: print("No of segments in the dataset is {}".format(len(self.segments_dict_list)))
        self.num_frames = 0
        for segment_dict in self.segments_dict_list:
            # add total number of frames in every segment
            self.num_frames += segment_dict['frame_count']
        # self.scan_names = sorted(list(set([os.path.basename(x).split("_")[1].split('.')[0] for x in os.listdir(os.path.join(self.data_path, 'training', 'votes'))])))
        self.num_points = num_points
        self.augment = augment
        self.use_height = use_height
       
    def __len__(self):
        return self.num_frames
    

    def resolve_idx_to_frame_path(self, idx):
        ''' Get Global idx and transorm into segment frame idx
        '''
        frame_idx = idx
        for segment_dict in self.segments_dict_list:
            if frame_idx >= segment_dict['frame_count']:
                frame_idx -= segment_dict['frame_count']
            else:
                frames_list = os.listdir(os.path.join(self.data_path, self.split_set, segment_dict['id']))
                frame_path = os.path.join(self.data_path, self.split_set, segment_dict['id'], frames_list[frame_idx])
                if not os.path.exists(frame_path):
                    raise ValueError("Frame path doesn't exist, error in idx_to_frame_path function")
                return frame_path


    def filtrate_objects(self, labels):
        '''
        obje_list Nx8 array contains all annotated objects
        '''
        type_whitelist = [self.class2type[i] for i in self.classes]
        
        # remove unwanted classes
        rows_to_be_deleted = []
        for i in range(labels.shape[0]):
            if not labels[i,0] in type_whitelist:
                rows_to_be_deleted.append(i)
        labels = np.delete(labels, rows_to_be_deleted, 0)
        return labels


    def __getitem__(self, idx):
        """
        Returns a dict with following keys:
            point_clouds: (N,3+C)
            center_label: (MAX_NUM_OBJ,3) for GT box center XYZ
            heading_class_label: (MAX_NUM_OBJ,) with int values in 0,...,NUM_HEADING_BIN-1
            heading_residual_label: (MAX_NUM_OBJ,)
            size_classe_label: (MAX_NUM_OBJ,) with int values in 0,...,NUM_SIZE_CLUSTER
            size_residual_label: (MAX_NUM_OBJ,3)
            sem_cls_label: (MAX_NUM_OBJ,) semantic class index
            box_label_mask: (MAX_NUM_OBJ) as 0/1 with 1 indicating a unique box
            vote_label: (N,9) with votes XYZ (3 votes: X1Y1Z1, X2Y2Z2, X3Y3Z3)
                if there is only one vote than X1==X2==X3 etc.
            vote_label_mask: (N,) with 0/1 with 1 indicating the point
                is in one of the object's OBB.
            scan_idx: int scan index in scan_names list
            max_gt_bboxes: unused
        """
        frame_data_path = self.resolve_idx_to_frame_path(idx)
        segment_id = frame_data_path.split('/')[-2] 
        frame_idx = frame_data_path.split('/')[-1].split('_')[-1].split('.')[0]
        # print('data idx is ', idx)
        # print('extracted segment id is ', segment_id)
        # print('extracted frame idx is ', frame_idx)
        # print("path is ", frame_data_path)
        
        point_cloud = np.load(os.path.join(self.data_path, self.split_set, 'votes', '{}'.format(segment_id), '{}_{}_pc.npz'.format(segment_id, frame_idx)))['pc'] # Nx3
        if not os.path.exists(os.path.join(self.data_path, self.split_set, 'votes', '{}'.format(segment_id), '{}_{}_pc.npz'.format(segment_id, frame_idx))):
            print('this path does not exist !!')
            print(os.path.join(self.data_path, self.split_set, 'votes', '{}'.format(segment_id), '{}_{}_pc.npz'.format(segment_id, frame_idx)))

        assert point_cloud.shape[1] == 3
        
        frame_data_path = os.path.join(self.data_path, self.split_set,'{}'.format(segment_id) ,'{}_{}.npz'.format(segment_id, frame_idx))
        
        frame_data = np.load(frame_data_path)
        labels = frame_data['labels']        
        assert labels.shape[1] == 8
        # print('labels types before filterations ', labels[:,0])
        labels = self.filtrate_objects(labels)
        # print('labels types after filterations ', labels[:,0])
        # create bboxes matrix
        bboxes = np.zeros_like(labels)
        for i in range(labels.shape[0]):
            # if labels[i,0] in self.excluded_labels: # skip signs and unknown labels
            #     continue
            bboxes[i, 0:3] = labels[i,4:7] #centers
            bboxes[i, 3:6] = labels[i,1:4] #lwh
            bboxes[i, 6] = labels[i,7] # heading
            bboxes[i, 7] = DC.raw2used_labels[labels[i,0]] #label
        
        
        point_votes = np.load(os.path.join(self.data_path, self.split_set, 'votes', '{}'.format(segment_id) ,'{}_{}_votes.npz'.format(segment_id, frame_idx)))['point_votes'] # Nx10

        assert point_votes.shape[1] == 10

        
        point_cloud = point_cloud[:,0:3]

        if self.use_height:
            floor_height = np.percentile(point_cloud[:,2],0.99)
            height = point_cloud[:,2] - floor_height
            point_cloud = np.concatenate([point_cloud, np.expand_dims(height, 1)],1) # (N,4)

        # ------------------------------- DATA AUGMENTATION ------------------------------
        if self.augment:
            raise NotImplementedError
            # Rotation along up-axis/Z-axis
            rot_angle = (np.random.random()*np.pi/3) - np.pi/6 # -30 ~ +30 degree
            rot_mat = waymo_utils.rotz(rot_angle)

            point_votes_end = np.zeros_like(point_votes)
            point_votes_end[:,1:4] = np.dot(point_cloud[:,0:3] + point_votes[:,1:4], np.transpose(rot_mat))
            point_votes_end[:,4:7] = np.dot(point_cloud[:,0:3] + point_votes[:,4:7], np.transpose(rot_mat))
            point_votes_end[:,7:10] = np.dot(point_cloud[:,0:3] + point_votes[:,7:10], np.transpose(rot_mat))

            point_cloud[:,0:3] = np.dot(point_cloud[:,0:3], np.transpose(rot_mat))
            bboxes[:,0:3] = np.dot(bboxes[:,0:3], np.transpose(rot_mat))
            bboxes[:,6] -= rot_angle
            point_votes[:,1:4] = point_votes_end[:,1:4] - point_cloud[:,0:3]
            point_votes[:,4:7] = point_votes_end[:,4:7] - point_cloud[:,0:3]
            point_votes[:,7:10] = point_votes_end[:,7:10] - point_cloud[:,0:3]

            

            # Augment point cloud scale: 0.85x-1.15x
            scale_ratio = np.random.random()*0.3+0.85
            scale_ratio = np.expand_dims(np.tile(scale_ratio,3),0)
            point_cloud[:,0:3] *= scale_ratio
            bboxes[:,0:3] *= scale_ratio
            bboxes[:,3:6] *= scale_ratio
            point_votes[:,1:4] *= scale_ratio
            point_votes[:,4:7] *= scale_ratio
            point_votes[:,7:10] *= scale_ratio
            if self.use_height:
                point_cloud[:,-1] *= scale_ratio[0,0]

        # ------------------------------- LABELS ------------------------------
        box3d_centers = np.zeros((MAX_NUM_OBJ, 3))
        box3d_sizes = np.zeros((MAX_NUM_OBJ, 3))
        angle_classes = np.zeros((MAX_NUM_OBJ,))
        angle_residuals = np.zeros((MAX_NUM_OBJ,))
        size_classes = np.zeros((MAX_NUM_OBJ,))
        size_residuals = np.zeros((MAX_NUM_OBJ, 3))
        label_mask = np.zeros((MAX_NUM_OBJ))
        label_mask[0:bboxes.shape[0]] = 1
        max_bboxes = np.zeros((MAX_NUM_OBJ, 8))
        max_bboxes[0:bboxes.shape[0],:] = bboxes

        for i in range(bboxes.shape[0]):
            bbox = bboxes[i]
            semantic_class = bbox[7]
            box3d_center = bbox[0:3]
            angle_class, angle_residual = DC.angle2class(bbox[6])
            # NOTE: The mean size stored in size2class is of full length of box edges,
            # while in sunrgbd_data.py data dumping we dumped *half* length l,w,h.. so have to time it by 2 here 
            box3d_size = bbox[3:6]
            size_class, size_residual = DC.size2class(box3d_size, DC.class2type[semantic_class])
            box3d_centers[i,:] = box3d_center
            angle_classes[i] = angle_class
            angle_residuals[i] = angle_residual
            size_classes[i] = size_class
            size_residuals[i] = size_residual
            box3d_sizes[i,:] = box3d_size

        target_bboxes_mask = label_mask 
        target_bboxes = np.zeros((MAX_NUM_OBJ, 6))
        for i in range(bboxes.shape[0]):
            bbox = bboxes[i]
            corners_3d = np.transpose(get_corners_from_labels_array(bbox)) # 8 x 3
            # import pdb; pdb.set_trace()
            # compute axis aligned box
            xmin = np.min(corners_3d[:,0])
            ymin = np.min(corners_3d[:,1])
            zmin = np.min(corners_3d[:,2])
            xmax = np.max(corners_3d[:,0])
            ymax = np.max(corners_3d[:,1])
            zmax = np.max(corners_3d[:,2])
            target_bbox = np.array([(xmin+xmax)/2, (ymin+ymax)/2, (zmin+zmax)/2, xmax-xmin, ymax-ymin, zmax-zmin])
            target_bboxes[i,:] = target_bbox

        point_cloud, choices = pc_util.random_sampling(point_cloud, self.num_points, return_choices=True)
        point_votes_mask = point_votes[choices,0]
        point_votes = point_votes[choices,1:]

        ret_dict = {}
        ret_dict['point_clouds'] = point_cloud.astype(np.float32)
        ret_dict['center_label'] = target_bboxes.astype(np.float32)[:,0:3]
        ret_dict['heading_class_label'] = angle_classes.astype(np.int64)
        ret_dict['heading_residual_label'] = angle_residuals.astype(np.float32)
        ret_dict['size_class_label'] = size_classes.astype(np.int64)
        ret_dict['size_residual_label'] = size_residuals.astype(np.float32)
        target_bboxes_semcls = np.zeros((MAX_NUM_OBJ))
        target_bboxes_semcls[0:bboxes.shape[0]] = bboxes[:,-1] # from 0 to 4
        ret_dict['sem_cls_label'] = target_bboxes_semcls.astype(np.int64)
        ret_dict['box_label_mask'] = target_bboxes_mask.astype(np.float32)
        ret_dict['vote_label'] = point_votes.astype(np.float32)
        ret_dict['vote_label_mask'] = point_votes_mask.astype(np.int64)
        # ret_dict['scan_idx'] = np.array(idx).astype(np.int64) # TODO: wrong indicator, add frame name and segment name instead
        # ret_dict['max_gt_bboxes'] = max_bboxes #ABAHNASY: not used parameter
        return ret_dict

def viz_votes(pc, point_votes, point_votes_mask):
    """ Visualize point votes and point votes mask labels
    pc: (N,3 or 6), point_votes: (N,9), point_votes_mask: (N,)
    """
    inds = (point_votes_mask==1)
    pc_obj = pc[inds,0:3]
    pc_obj_voted1 = pc_obj + point_votes[inds,0:3]
    pc_obj_voted2 = pc_obj + point_votes[inds,3:6]
    pc_obj_voted3 = pc_obj + point_votes[inds,6:9]
    pc_util.write_ply(pc_obj, 'pc_obj.ply')
    pc_util.write_ply(pc_obj_voted1, 'pc_obj_voted1.ply')
    pc_util.write_ply(pc_obj_voted2, 'pc_obj_voted2.ply')
    pc_util.write_ply(pc_obj_voted3, 'pc_obj_voted3.ply')

def viz_obb(pc, label, mask, angle_classes, angle_residuals,
    size_classes, size_residuals):
    """ Visualize oriented bounding box ground truth
    pc: (N,3)
    label: (K,3)  K == MAX_NUM_OBJ
    mask: (K,)
    angle_classes: (K,)
    angle_residuals: (K,)
    size_classes: (K,)
    size_residuals: (K,3)
    """
    oriented_boxes = []
    K = label.shape[0]
    for i in range(K):
        if mask[i] == 0: continue
        obb = np.zeros(7)
        obb[0:3] = label[i,0:3]
        heading_angle = DC.class2angle(angle_classes[i], angle_residuals[i])
        box_size = DC.class2size(size_classes[i], size_residuals[i])
        obb[3:6] = box_size
        obb[6] = -1 * heading_angle
        print(obb)
        oriented_boxes.append(obb)
    pc_util.write_oriented_bbox(oriented_boxes, 'gt_obbs.ply')
    pc_util.write_ply(label[mask==1,:], 'gt_centroids.ply')

def get_sem_cls_statistics():
    """ Compute number of objects for each semantic class """
    d = WaymoDetectionVotesDataset(use_height=True, augment=False)
    sem_cls_cnt = {}
    for i in range(len(d)):
        if i%10==0: print(i)
        sample = d[i]
        pc = sample['point_clouds']
        sem_cls = sample['sem_cls_label']
        mask = sample['box_label_mask']
        for j in sem_cls:
            if mask[j] == 0: continue
            if sem_cls[j] not in sem_cls_cnt:
                sem_cls_cnt[sem_cls[j]] = 0
            sem_cls_cnt[sem_cls[j]] += 1
    print(sem_cls_cnt)

if __name__=='__main__':
    d = WaymoDetectionVotesDataset(use_height=True, augment=False)
    # for i in range(len(d)):
    
    sample = d[0]
    print(sample['vote_label'].shape, sample['vote_label_mask'].shape)
    pc_util.write_ply(sample['point_clouds'], 'pc.ply')
    viz_votes(sample['point_clouds'], sample['vote_label'], sample['vote_label_mask'])
    viz_obb(sample['point_clouds'], sample['center_label'], sample['box_label_mask'],
        sample['heading_class_label'], sample['heading_residual_label'],
        sample['size_class_label'], sample['size_residual_label'])
