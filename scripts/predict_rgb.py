import argparse
import os
import shutil
import sys
import time
import re
import numpy as np
import torch
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
import torch.nn as nn
from pathlib import Path

try:
    import laspy
except ImportError:
    laspy = None

# Pastikan path modul terbaca
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)                    # data_utils/ and models/ inside scripts/
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))   # or one level above scripts/

from data_utils.dataLoader import ScannetDatasetWholeScene
from models.dgcnn_sem_seg import dgcnn_sem_seg
from data_utils.split_merge_las import calculate_block_size, split_indices

def read_las(las_file):
    inFile = laspy.read(las_file)
    r = np.asarray(inFile.red, dtype=np.float64)
    g = np.asarray(inFile.green, dtype=np.float64)
    b = np.asarray(inFile.blue, dtype=np.float64)
    # Same RGB scaling as 01_data_conversion.py (16-bit -> 0-255)
    if max(r.max(), g.max(), b.max()) > 255:
        r, g, b = r / 256.0, g / 256.0, b / 256.0
    data = np.column_stack((inFile.x, inFile.y, inFile.z, r, g, b))
    return data, inFile

def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size in testing [default: 32]')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--num_point', type=int, default=4096, help='point number [default: 4096]')
    parser.add_argument('--model', type=str, required=True, help='DGCNN best checkpoint model')
    parser.add_argument('--num_votes', type=int, default=1, help='aggregate segmentation scores with voting [default: 1 for speed]')
    parser.add_argument('--num_classes', type=int, default=3, help='How many classes used for segmentation')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N', help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=20, metavar='N', help='Num of nearest neighbors to use')
    parser.add_argument('--point_cloud', type=str, required=True, help='Name of point cloud data')
    parser.add_argument('--block_size', type=int, default=500, help='Size of each block')
    parser.add_argument('--output_dir', type=str, required=True, help='Complete path to the output LAS file')
    return parser.parse_args()

def add_vote(vote_label_pool, point_idx, pred_label, weight):
    valid = (weight != 0) & np.isfinite(weight) & (pred_label < vote_label_pool.shape[1])
    np.add.at(vote_label_pool,
              (point_idx[valid].astype(np.int64), pred_label[valid].astype(np.int64)), 1)
    return vote_label_pool

def process_area(args, classifier, block_data, room_max):
    """
    Fungsi ini memproses satu area/blok. Model sudah dimuat sebelumnya.
    Mengembalikan array numpy hasil prediksi tanpa menyimpan ke disk.
    """
    NUM_CLASSES = args.num_classes
    BATCH_SIZE = args.batch_size
    NUM_POINT = args.num_point
    root = 'data/sem_seg_data/'

    TEST_DATASET_WHOLE_SCENE = ScannetDatasetWholeScene(data_list=[block_data], block_points=NUM_POINT, room_max=room_max)

    with torch.no_grad():
        num_batches = 1 
        for batch_idx in range(num_batches):
            whole_scene_data = TEST_DATASET_WHOLE_SCENE.scene_points_list[batch_idx]
            whole_scene_label = TEST_DATASET_WHOLE_SCENE.semantic_labels_list[batch_idx]
            vote_label_pool = np.zeros((whole_scene_label.shape[0], NUM_CLASSES))
            
            scene_data, scene_label, scene_smpw, scene_point_index = TEST_DATASET_WHOLE_SCENE[batch_idx]
            num_blocks_in_scene = scene_data.shape[0]
            s_batch_num = (num_blocks_in_scene + BATCH_SIZE - 1) // BATCH_SIZE
            
            batch_data = np.zeros((BATCH_SIZE, NUM_POINT, 9)) 
            batch_label = np.zeros((BATCH_SIZE, NUM_POINT))
            batch_point_index = np.zeros((BATCH_SIZE, NUM_POINT))
            batch_smpw = np.zeros((BATCH_SIZE, NUM_POINT))

            for sbatch in range(s_batch_num):
                start_idx = sbatch * BATCH_SIZE
                end_idx = min((sbatch + 1) * BATCH_SIZE, num_blocks_in_scene)
                real_batch_size = end_idx - start_idx
                
                batch_data[0:real_batch_size, ...] = scene_data[start_idx:end_idx, ...]
                batch_label[0:real_batch_size, ...] = scene_label[start_idx:end_idx, ...]
                batch_point_index[0:real_batch_size, ...] = scene_point_index[start_idx:end_idx, ...]
                batch_smpw[0:real_batch_size, ...] = scene_smpw[start_idx:end_idx, ...]

                torch_data = torch.Tensor(batch_data).float().cuda()
                torch_data = torch_data.transpose(2, 1)

                seg_pred = classifier(torch_data)
                seg_pred = seg_pred.permute(0, 2, 1).contiguous()
                pred = seg_pred.max(dim=2)[1]
                pred_np = pred.detach().cpu().numpy()
                
                vote_label_pool = add_vote(vote_label_pool, batch_point_index[0:real_batch_size, ...],
                                           pred_np[0:real_batch_size, ...],
                                           batch_smpw[0:real_batch_size, ...])
                
            pred_label = np.argmax(vote_label_pool, 1)
            return pred_label

if __name__ == '__main__':
    start_time = time.time()
    args = parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
   # 1. BACA & SPLIT DATA LANGSUNG DI RAM
    print("Preparing data and creating temporary block...", flush=True)
    data, las_in = read_las(args.point_cloud)
    xy_min = np.amin(data, axis=0)[0:2]
    data[:, 0:2] -= xy_min                # same shift as 01_data_conversion.py
    room_max = data[:, 0:3].max(axis=0)   # same as max_room_x/y/z in training

    dummy_label = np.ones((data.shape[0], 1))
    data = np.hstack((data, dummy_label))

    num_blocks_x, num_blocks_y = calculate_block_size(data, args.block_size)
    block_indices = split_indices(data, num_blocks_x, num_blocks_y, args.block_size)
    
    # 2. LOAD MODEL HANYA SEKALI DI SINI
    print("Load DGCNN model to GPU...", flush=True)
    classifier = dgcnn_sem_seg(args).cuda()
    classifier = nn.DataParallel(classifier)
    checkpoint = torch.load(args.model, weights_only=True, mmap=True)
    classifier.load_state_dict(checkpoint)
    classifier = classifier.eval()
    
    # 3. PROSES INFERENCE (Super Cepat)
    print("Start Classification...", flush=True)
    full_pred = np.zeros(data.shape[0], dtype=np.int64)
    total_areas = len(block_indices)

    for idx, pidx in enumerate(block_indices):
        if len(pidx) > 0:
            full_pred[pidx] = process_area(args, classifier, data[pidx], room_max)
        print(f"PROGRESS:{idx + 1}/{total_areas}", flush=True)
        
    # 4. GABUNGKAN DATA & EXPORT .LAS SEKALI SAJA
    print("Writing classification into the original point cloud...")
    os.makedirs(args.output_dir, exist_ok=True)
    filename = os.path.basename(str(args.point_cloud))
    name, ext = os.path.splitext(filename)
    out_las = os.path.join(args.output_dir, f"{name}_classified{ext}")

    map_array = np.array([2, 5, 6], dtype=np.uint8)   # 0->2, 1->5, 2->6
    las_in.classification = map_array[full_pred]
    las_in.write(out_las)
    print(f"Classified point cloud saved in: {out_las}")

    # 5. CLEAN UP FOLDER SEMENTARA
    # shutil.rmtree(data_dir, ignore_errors=True)
    
    elapsed_time = time.time() - start_time
    print('Classification Process Completed!')
    print(f"Elapsed Time: {elapsed_time:.2f} seconds")