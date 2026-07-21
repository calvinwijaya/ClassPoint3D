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
BASE_DIR = os.path.dirname(os.path.abspath(os.getcwd()))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

from data_utils.dataLoader import ScannetDatasetWholeScene
from models.dgcnn_sem_seg import dgcnn_sem_seg
from data_utils.split_merge_las import calculate_block_size, split_array

classes = ['0', '1', '2']
class2label = {cls: i for i, cls in enumerate(classes)}

def read_las(las_files):
    inFile = laspy.read(las_files)
    x = inFile.x
    y = inFile.y
    z = inFile.z
    r = inFile.red
    g = inFile.green
    b = inFile.blue
    data = np.column_stack((x, y, z, r, g, b))
    return data

def save_las(X, filename):
    header = laspy.LasHeader(point_format=2, version="1.2")
    las = laspy.LasData(header)
    las.x = X[:, 0]
    las.y = X[:, 1]
    las.z = X[:, 2]
    las.red = X[:, 3]
    las.green = X[:, 4]
    las.blue = X[:, 5]
    
    # Mapping otomatis (0->2, 1->5, 2->6)
    map_array = np.array([2, 5, 6]) 
    mapped_class = map_array[X[:, 6].astype(int)]
    las.classification = mapped_class.astype(np.uint8)
    las.write(filename)

def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size in testing [default: 32]')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--num_point', type=int, default=4096, help='point number [default: 4096]')
    parser.add_argument('--model', type=str, required=True, help='DGCNN best checkpoint model')
    parser.add_argument('--test_area', type=int, default=5, help='area for testing, option: 1-6')
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
    B = pred_label.shape[0]
    N = pred_label.shape[1]
    for b in range(B):
        for n in range(N):
            if weight[b, n] != 0 and not np.isinf(weight[b, n]):
                if int(pred_label[b, n]) < vote_label_pool.shape[1]:
                    vote_label_pool[int(point_idx[b, n]), int(pred_label[b, n])] += 1
    return vote_label_pool

def process_area(args, classifier, test_area, xy_min, block_data):
    """
    Fungsi ini memproses satu area/blok. Model sudah dimuat sebelumnya.
    Mengembalikan array numpy hasil prediksi tanpa menyimpan ke disk.
    """
    NUM_CLASSES = args.num_classes
    BATCH_SIZE = args.batch_size
    NUM_POINT = args.num_point
    root = 'data/sem_seg_data/'

    # TEST_DATASET_WHOLE_SCENE = ScannetDatasetWholeScene(root, split='test', test_area=test_area, block_points=NUM_POINT)
    TEST_DATASET_WHOLE_SCENE = ScannetDatasetWholeScene(data_list=[block_data], block_points=NUM_POINT)

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
                batch_data[:, :, 3:6] /= 1.0 

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
            whole_scene_data = whole_scene_data.astype(np.float64)
            
            # Kembalikan ke koordinat UTM asli
            whole_scene_data[:, 0:2] += xy_min

            # --- OPTIMASI: Vectorization (Hindari for-loop untuk data besar) ---
            pred_label_col = pred_label.reshape(-1, 1)
            # Ambil hanya X, Y, Z, R, G, B (indeks 0 sampai 5), abaikan dummy label di kolom 6
            result_data = np.hstack((whole_scene_data[:, 0:6], pred_label_col))
            
            return result_data

if __name__ == '__main__':
    start_time = time.time()
    args = parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
   # 1. BACA & SPLIT DATA LANGSUNG DI RAM
    print("Mempersiapkan data dan membuat blok memori sementara...", flush=True)
    data = read_las(args.point_cloud)
    xy_min = np.amin(data, axis=0)[0:2]
    data[:, 0:2] -= xy_min
    
    zeros_column = np.ones((data.shape[0], 1))
    data = np.hstack((data, zeros_column))
        
    num_blocks_x, num_blocks_y = calculate_block_size(data, args.block_size)
    blocks = split_array(data, num_blocks_x, num_blocks_y, args.block_size)
    
    # 2. LOAD MODEL HANYA SEKALI DI SINI
    print("Memuat Model DGCNN ke GPU...", flush=True)
    classifier = dgcnn_sem_seg(args).cuda()
    classifier = nn.DataParallel(classifier)
    checkpoint = torch.load(args.model, weights_only=True, mmap=True)
    classifier.load_state_dict(checkpoint)
    classifier = classifier.eval()
    
    # 3. PROSES INFERENCE (Super Cepat)
    print("Memulai klasifikasi...", flush=True)
    all_results = []
    total_areas = len(blocks)
    
    # Langsung lakukan loop pada array 'blocks' yang ada di RAM
    for idx, block_data in enumerate(blocks):
        # Lewati blok kosong atau yang terlalu kecil (jika perlu)
        if block_data.nbytes < 100: continue 

        args.test_area = idx
        block_result = process_area(args, classifier, idx, xy_min, block_data)
        all_results.append(block_result)
        
        # Kirim sinyal progres ke GUI
        print(f"PROGRESS:{idx + 1}/{total_areas}", flush=True)
        
    # 4. GABUNGKAN DATA & EXPORT .LAS SEKALI SAJA
    print("Menggabungkan hasil klasifikasi dan mengekspor file LAS...")
    if all_results:
        # Menyatukan seluruh blok yang ada di memori menggunakan numpy
        final_point_cloud = np.vstack(all_results)
        
        filename = os.path.basename(str(args.point_cloud))
        name, ext = os.path.splitext(filename)
        out_las = os.path.join(args.output_dir, f"{name}_classified.las")
        
        # Simpan satu kali langsung jadi utuh
        save_las(final_point_cloud, out_las)
        print(f"File berhasil disimpan di: {out_las}")
    else:
        print("Peringatan: Tidak ada blok data valid yang diproses.")

    # 5. CLEAN UP FOLDER SEMENTARA
    # shutil.rmtree(data_dir, ignore_errors=True)
    
    elapsed_time = time.time() - start_time
    print('Classification Process Completed!')
    print(f"Elapsed Time: {elapsed_time:.2f} seconds")