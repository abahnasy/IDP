""" Waymo dataset for segmentation.

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
sys.path.append(BASE_DIR)
import utils

RAW_LABELS =  {0: 'TYPE_UNKNOWN', 1: 'TYPE_VEHICLE' , 2: 'TYPE_PEDESTRIAN', 3: 'TYPE_SIGN', 4: 'TYPE_CYCLIST'}

class WaymoSegmentationDataset(Dataset):
    def __init__(self, data_path, split_set='train', num_points=180000,
        use_height=False,
        augment=False,
        verbose:bool = True):
        
        self.type2class = {0: 'TYPE_UNKNOWN', 1: 'TYPE_VEHICLE' , 2: 'TYPE_PEDESTRIAN', 3: 'TYPE_SIGN', 4: 'TYPE_CYCLIST'}
        self.class2type = {self.type2class[t]:t for t in self.type2class}
        self.classes = ['TYPE_VEHICLE']
        # self.mapping_labels = {1:0,2:1,4:2} # map dataset labels to our labels to handle discarded classes 
        # self.excluded_labels = [0,3] # exclude unknowns and signs labels
        self.split_set = split_set
        self.data_path = data_path

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
        """
        frame_data_path = self.resolve_idx_to_frame_path(idx)
        # print("loaded frame is {}".format(frame_data_path))
        segment_id = frame_data_path.split('/')[-2] 
        frame_idx = frame_data_path.split('/')[-1].split('_')[-1].split('.')[0]
        
        frame_data = np.load(frame_data_path)
        point_cloud = frame_data['pc']
        # print("original pc size is {}".format(point_cloud.shape))
        assert point_cloud.shape[1] == 3
        # sample pc to fixed size
        try:
            point_cloud = utils.random_sampling(point_cloud, self.num_points)
        except:
            print("defected frame ", frame_data_path)
            
        assert point_cloud.shape[0] == self.num_points
        labels = frame_data['labels'] # box = [label, length, width, height, x, y, z, heading]
        
        
        assert labels.shape[1] == 8
        labels = self.filtrate_objects(labels)
        
        
        # create bboxes matrix
        bboxes = np.zeros_like(labels)
        for i in range(labels.shape[0]): # [x,y,z,l,w,h,heading, label]
            bboxes[i, 0:3] = labels[i,4:7] #centers
            bboxes[i, 3:6] = labels[i,1:4] #lwh
            bboxes[i, 6] = labels[i,7] # heading
            bboxes[i, 7] = labels[i,0] #label
        
        
        point_cloud = point_cloud[:,0:3]
        point_segmentation = np.zeros((self.num_points), dtype=np.int32)

        if self.use_height:
            floor_height = np.percentile(point_cloud[:,2],0.99)
            height = point_cloud[:,2] - floor_height
            point_cloud = np.concatenate([point_cloud, np.expand_dims(height, 1)],1) # (N,4)
        
        # get foreground/background segmentations
        extend_gt_boxes3d = utils.enlarge_box3d(bboxes, extra_width=0.2)
        
        for i in range(bboxes.shape[0]):
            try:
                # Find all points in this object's OBB
                box3d_pts_3d = np.transpose(utils.get_corners_from_labels_array(bboxes[i,:]))
                # if verbose: print(" 3d box data shape {}".format(box3d_pts_3d.shape))
                # if verbose: print("Box coordinates of the current object are {}".format(box3d_pts_3d))
                pc_in_box3d,inds = utils.extract_pc_in_box3d(point_cloud, box3d_pts_3d)
                # if verbose: print("list of indices inside the box {}".format(inds))
                # if verbose: print("No. of points inside the box are {}".format(len(pc_in_box3d)))
                # Assign first dimension to indicate it is in an object box
                point_segmentation[inds] = 1
                # enlarge the bbox3d, ignore nearby points
                extend_gt_corners = np.transpose(utils.get_corners_from_labels_array(extend_gt_boxes3d[i,:]))
                pc_in_box3d,inds_extended = utils.extract_pc_in_box3d(point_cloud, extend_gt_corners)

                ignore_flag = np.logical_xor(inds, inds_extended)
                point_segmentation[ignore_flag] = -1
            except Exception as e:
                print(e)
                # print('ERROR, idx {}, classlabel {} and not found in whitelist'.format(data_idx, obj.label))
                raise    

        ret_dict = {}
        ret_dict['point_clouds'] = point_cloud.astype(np.float32)
        # ret_dict['oriented_bboxes'] = bboxes
        ret_dict['point_label'] = point_segmentation.astype(np.int64)
        # print("sum of gt points from dataloader is ", ret_dict['point_label'].sum())
        return ret_dict


    def collate_batch(self, batch):
        batch_size = batch.__len__()
        ans_dict = {}

        for key in batch[0].keys():
            if isinstance(batch[0][key], np.ndarray):
                ans_dict[key] = np.concatenate([batch[k][key][np.newaxis, ...] for k in range(batch_size)], axis=0)

            else:
                ans_dict[key] = [batch[k][key] for k in range(batch_size)]
                if isinstance(batch[0][key], int):
                    ans_dict[key] = np.array(ans_dict[key], dtype=np.int32)
                elif isinstance(batch[0][key], float):
                    ans_dict[key] = np.array(ans_dict[key], dtype=np.float32)

        return ans_dict
    
    
    def count_mean_points_inside_bbox(self):
        num_pts_list = []
        for idx in range(self.__len__()):
            # print('processing frame {}'.format(idx))
            frame_data_path = self.resolve_idx_to_frame_path(idx)
            # print("loaded frame is {}".format(frame_data_path))
            # segment_id = frame_data_path.split('/')[-2] 
            # frame_idx = frame_data_path.split('/')[-1].split('_')[-1].split('.')[0]
            
            frame_data = np.load(frame_data_path)
            point_cloud = frame_data['pc']
            labels = frame_data['labels'] # box = [label, length, width, height, x, y, z, heading]
            labels = self.filtrate_objects(labels)
        
        
            # create bboxes matrix
            bboxes = np.zeros_like(labels)
            for i in range(labels.shape[0]): # [x,y,z,l,w,h,heading, label]
                bboxes[i, 0:3] = labels[i,4:7] #centers
                bboxes[i, 3:6] = labels[i,1:4] #lwh
                bboxes[i, 6] = labels[i,7] # heading
                bboxes[i, 7] = labels[i,0] #label

            for i in range(bboxes.shape[0]):
                box3d_pts_3d = np.transpose(utils.get_corners_from_labels_array(bboxes[i,:]))
                # if verbose: print(" 3d box data shape {}".format(box3d_pts_3d.shape))
                # if verbose: print("Box coordinates of the current object are {}".format(box3d_pts_3d))
                pc_in_box3d,inds = utils.extract_pc_in_box3d(point_cloud, box3d_pts_3d)
                if pc_in_box3d.shape[0] > 50000:
                    print(idx)    
                num_pts_list.append(pc_in_box3d.shape[0])

        print('mean_points in bbox is {}'.format(np.mean(num_pts_list)))
        import pickle
        filehandler = open(b"vechicle_num_pts.obj","wb")
        pickle.dump(num_pts_list,filehandler)
        print('File is saved on desk !')





if __name__ == '__main__':

    # train_set = WaymoSegmentationDataset('/home/bahnasy/IDP/waymo_open_dataset/dataset', split_set='train')
    # train_set.count_mean_points_inside_bbox()
    train_set = WaymoSegmentationDataset('/home/bahnasy/IDP/waymo_open_dataset/dataset', split_set='train')
    sample = train_set[6575]
    utils.write_ply(sample['point_clouds'], 'pc.ply')
    mask = sample['point_label'].astype(np.bool)
    # mask = np.squeeze(sample['point_label'].astype(np.bool) ,1)
    print('point labels shape', sample['point_label'].astype(np.bool).shape)
    print(mask.sum())
    pc = sample['point_clouds']
    oriented_boxes = sample['oriented_bboxes']
    print(pc[mask, :].shape)
    utils.write_ply(pc[mask], 'fg_seg.ply')
    utils.write_oriented_bbox(oriented_boxes, 'gt_obbs.ply')

    # train_set = WaymoSegmentationDataset('/home/bahnasy/IDP/waymo_open_dataset/dataset', split_set='train')
    # from torch.utils.data import DataLoader
    # train_loader = DataLoader(train_set, batch_size=1, shuffle=True, pin_memory=True, num_workers=0, collate_fn=train_set.collate_batch)
    # for it, batch in enumerate(train_loader):
    #     pts_input, cls_labels, oriented_boxes = batch['point_clouds'], batch['point_label'], batch['oriented_bboxes']
    #     for b in range(pts_input.shape[0]): # b for batch_size:
    #         pc = pts_input[b]
    #         mask = np.squeeze(cls_labels[b].astype(np.bool) ,1)
    #         bboxes = oriented_boxes[b]
            
    #         utils.write_ply(pc, 'pc_{}.ply'.format(b))
    #         utils.write_ply(pc[mask], 'fg_seg_{}.ply'.format(b))
    #         utils.write_oriented_bbox(bboxes, 'gt_obbs_{}.ply'.format(b))
            
            
    
    # exit()
    
    # print(len(ds))
    # sample = ds[5]
    # # utils.write_ply(sample['point_clouds'], 'pc.ply')
    # # mask = sample['point_label'].sum()
    # mask = np.squeeze(sample['point_label'].astype(np.bool) ,1)
    # print(mask.shape)
    # # print(mask.sum())
    # pc = sample['point_clouds']
    # oriented_boxes = sample['oriented_bboxes']
    # print(pc[mask, :].shape)
    # # utils.write_ply(pc[mask], 'fg_seg.ply')
    # utils.write_oriented_bbox(oriented_boxes, 'gt_obbs.ply')
