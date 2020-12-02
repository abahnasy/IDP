### Train on your own data

[For Pro Users] If you have your own dataset with point clouds and annotated 3D bounding boxes, you can create a new dataset class and train VoteNet on your own data. To ease the proces, some tips are provided below.

Firstly, you need to store point clouds in the upright coordinate system (Z is up, Y is forward, X is right-ward) and 3D bounding boxes as its center (x,y,z), size (l,w,h) and heading angle (along the up-axis; rotation radius from +X towards -Y; +X is 0 and -Y is pi/4). You can refer to `sunrgbd/sunrgbd_data.py` as to how to compute the groundtruth votes (translational vectors from object points to 3D bounding box centers). If your dataset has instance segmentation annotation, you can also compute groundtruth votes on the fly in the dataset class -- refer to `scannet/batch_load_scannet_data.py` and `scannet/scannet_detection_dataset.py` for more details.

Secondly, you need to create a new dataset class as well as to specify some config information about the dataset. For config information, you can refer to `sunrgbd/model_util_config.py` as an example and modify the `num_classes`, `type2class`, `num_size_clusters`, `mean_size_arr` etc. The `mean_size_arr` is computed by going through all 3D bounding boxes in the train set and cluster them (either by geometric size or semantic class) into several clusters and then compute the median box size in each cluster (an example porcess is [here](https://github.com/facebookresearch/votenet/blob/7c19af314a3d12532dc3c8dbd05d1d404c75891e/sunrgbd/sunrgbd_data.py#L264)). In both SUN RGB-D and ScanNet, we only consider one tempalte box size for each semantic class, but you can have multiple size templates for each class too (in which case you also need to modify the `size2class` function in the config). For detection dataset class, you can refer to `sunrgbd/sunrgbd_detection_dataset.py` and modify based on it. The major thing to modify is the dataset paths (in `__init__` function) and data loading methods (at the beginning of the `__getitem__` function), which depend on where and how you store the data.

Lastly, after you make sure the dataset class returns the correct input point clouds and ground truth labels, you need to add the new dataset to the `train.py` file and `eval.py` file by augmenting the options of `FLAGS.dataset` argument (adding another `elif` to the dataset set up section). Then by selecting your new dataset in `train.py`, you should be able to train a VoteNet on your own data!

Note that the VoteNet was originally used on SUN RGB-D and ScanNet which only have either 1D or 0D rotations in their annotated bounding boxes. It is possible to extend the VoteNet to predict 3D rotations though. One simple way is to supervise the network to predict three Euler angles. To support it you will need to prepare ground truth labels and then change the prediction of the 1D `heading_angle` to prediction of three Euler angles in the network output; and modify the 3D bounding box parameterization and transformations accordingly.

Feel free to post an issue if you meet any difficulty during the process!