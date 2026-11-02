import os
import sys
import glob
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
import h5py
import torch
import json
import cv2
import pickle

_H5_CACHE = {}

##########################################################################
# Add new classes for loading data from h5 files
def load_data_semseg(partition, test_area):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    # download_S3DIS()
    # prepare_test_data_semseg()
    DATA_DIR = 'data/'
    if partition == 'train':
        data_dir = os.path.join(DATA_DIR, 'sem_seg_hdf5_data')
    else:
        data_dir = os.path.join(DATA_DIR, 'sem_seg_hdf5_data')
    with open(os.path.join(data_dir, "all_files.txt")) as f:
        all_files = [line.rstrip() for line in f]
    with open(os.path.join(data_dir, "room_filelist.txt")) as f:
        room_filelist = [line.rstrip() for line in f]
    if 'all' in _H5_CACHE:
        data_batches, seg_batches = _H5_CACHE['all']
    else:
        data_batchlist, label_batchlist = [], []
        for f in all_files:
            with h5py.File(os.path.join(DATA_DIR, f), 'r') as file:
                data_batchlist.append(file["data"][:])
                label_batchlist.append(file["label"][:])
        data_batches = np.concatenate(data_batchlist, 0)
        seg_batches = np.concatenate(label_batchlist, 0)
        del data_batchlist, label_batchlist
        _H5_CACHE['all'] = (data_batches, seg_batches)
    test_area_name = "Area_" + test_area
    train_idxs, test_idxs = [], []
    for i, room_name in enumerate(room_filelist):
        if room_name == test_area_name:
            test_idxs.append(i)
        else:
            train_idxs.append(i)
    if partition == 'train':
        all_data = data_batches[train_idxs, ...]
        all_seg = seg_batches[train_idxs, ...]
    else:
        all_data = data_batches[test_idxs, ...]
        all_seg = seg_batches[test_idxs, ...]    
    return all_data, all_seg

class S3DIS(Dataset):
    def __init__(self, num_points=4096, partition='train', test_area='1'):
        self.data, self.seg = load_data_semseg(partition, test_area)
        self.num_points = num_points
        self.partition = partition
        # self.semseg_colors = load_color_semseg() 

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        seg = self.seg[item][:self.num_points]
        if self.partition == 'train':
            indices = list(range(pointcloud.shape[0]))
            np.random.shuffle(indices)
            pointcloud = pointcloud[indices]
            seg = seg[indices]

            theta = np.random.uniform(0, 2 * np.pi)
            c, s = np.cos(theta), np.sin(theta)
            xy = pointcloud[:, 0:2].copy()
            pointcloud[:, 0] = c * xy[:, 0] - s * xy[:, 1]
            pointcloud[:, 1] = s * xy[:, 0] + c * xy[:, 1]
        seg = torch.LongTensor(seg)
        return pointcloud, seg

    def __len__(self):
        return self.data.shape[0]

# Define class for testing set data loader
class ScannetDatasetWholeScene():
    # prepare to give prediction on each points
    def __init__(self, root=None, block_points=4096, split='test', test_area=5, stride=25.0, block_size=25.0,
                padding=0.001, data_list=None, room_max=None):
        self.block_points = block_points
        self.block_size = block_size
        self.padding = padding
        self.root = root
        self.split = split
        self.stride = stride
        self.room_max = room_max
        self.scene_points_num = []
        assert split in ['train', 'test']
        if data_list is not None:
            self.file_list = ['in_memory_%d' % i for i in range(len(data_list))]
            arrays = data_list
        else:
            if self.split == 'train':
                self.file_list = [d for d in os.listdir(root) if d.find('Area_%d' % test_area) == -1]
            else:
                self.file_list = [d for d in os.listdir(root) if d.find('Area_%d' % test_area) != -1]
            arrays = [np.load(root + f) for f in self.file_list]
        self.scene_points_list = []
        self.semantic_labels_list = []
        self.room_coord_min, self.room_coord_max = [], []
        for data in arrays:
            points = data[:, :3]
            self.scene_points_list.append(data[:, :-1])
            self.semantic_labels_list.append(data[:, -1])
            coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
            self.room_coord_min.append(coord_min), self.room_coord_max.append(coord_max)
        assert len(self.scene_points_list) == len(self.semantic_labels_list)

        labelweights = np.zeros(3) # Number of classes
        for seg in self.semantic_labels_list:
            tmp, _ = np.histogram(seg, range(4)) # Number of classes + 1
            self.scene_points_num.append(seg.shape[0])
            labelweights += tmp
        labelweights = labelweights.astype(np.float32)
        labelweights = labelweights / np.sum(labelweights)
        self.labelweights = np.power(np.amax(labelweights) / labelweights, 1 / 3.0)

    def __getitem__(self, index):
        point_set_ini = self.scene_points_list[index]
        points = point_set_ini[:,:178]
        labels = self.semantic_labels_list[index]
        coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
        grid_x = int(np.ceil(float(coord_max[0] - coord_min[0] - self.block_size) / self.stride) + 1)
        grid_y = int(np.ceil(float(coord_max[1] - coord_min[1] - self.block_size) / self.stride) + 1)
        data_parts, label_parts, weight_parts, index_parts = [], [], [], []
        for index_y in range(0, grid_y):
            for index_x in range(0, grid_x):
                s_x = coord_min[0] + index_x * self.stride
                e_x = min(s_x + self.block_size, coord_max[0])
                s_x = e_x - self.block_size
                s_y = coord_min[1] + index_y * self.stride
                e_y = min(s_y + self.block_size, coord_max[1])
                s_y = e_y - self.block_size
                point_idxs = np.where(
                    (points[:, 0] >= s_x - self.padding) & (points[:, 0] <= e_x + self.padding) & (points[:, 1] >= s_y - self.padding) & (
                                points[:, 1] <= e_y + self.padding))[0]
                if point_idxs.size == 0:
                    continue
                num_batch = int(np.ceil(point_idxs.size / self.block_points))
                point_size = int(num_batch * self.block_points)
                replace = False if (point_size - point_idxs.size <= point_idxs.size) else True
                point_idxs_repeat = np.random.choice(point_idxs, point_size - point_idxs.size, replace=replace)
                point_idxs = np.concatenate((point_idxs, point_idxs_repeat))
                np.random.shuffle(point_idxs)
                data_batch = points[point_idxs, :]
                normlized_xyz = np.zeros((point_size, 3))
                norm_max = self.room_max if self.room_max is not None else coord_max
                normlized_xyz[:, 0] = data_batch[:, 0] / norm_max[0]
                normlized_xyz[:, 1] = data_batch[:, 1] / norm_max[1]
                normlized_xyz[:, 2] = data_batch[:, 2] / norm_max[2]
                data_batch[:, 0] = data_batch[:, 0] - (s_x + self.block_size / 2.0)
                data_batch[:, 1] = data_batch[:, 1] - (s_y + self.block_size / 2.0)

                data_batch[:, 2] = data_batch[:, 2] - data_batch[:, 2].min()

                data_batch[:,3:6] /= 255 # Normalize color
                
                data_batch = np.concatenate((data_batch, normlized_xyz), axis=1)
                label_batch = labels[point_idxs].astype(int)
                batch_weight = self.labelweights[label_batch]

                data_parts.append(data_batch)
                label_parts.append(label_batch)
                weight_parts.append(batch_weight)
                index_parts.append(point_idxs)
        data_room = np.concatenate(data_parts, 0).reshape((-1, self.block_points, data_parts[0].shape[1]))
        label_room = np.concatenate(label_parts).reshape((-1, self.block_points))
        sample_weight = np.concatenate(weight_parts).reshape((-1, self.block_points))
        index_room = np.concatenate(index_parts).reshape((-1, self.block_points))
        return data_room, label_room, sample_weight, index_room

    def __len__(self):
        return len(self.scene_points_list)