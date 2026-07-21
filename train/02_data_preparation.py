import os
import numpy as np
import h5py
from data_utils.indoor3d_util import room2blocks_wrapper_normalized
import argparse

NUM_POINT = 4096 
H5_BATCH_SIZE = 1000
block_size = 25
data_dim = [NUM_POINT, 9] # XYZ, RGB, normXYZ
label_dim = [NUM_POINT]
stride = 5 # Overlap between blocks
sample_num = 10000
random_sample = False
data_dtype = 'float32'
label_dtype = 'uint8'
output_h5 = 'data/sem_seg_hdf5_data'

# Initialize buffer_size as a global variable
buffer_size = 0
h5_index = 0
h5_batch_data = None
h5_batch_label = None

def main():
    global buffer_size, h5_batch_data, h5_batch_label, h5_index
    data_dir = 'data/sem_seg_data'

    # Write numpy array data and label to h5_filename
    def save_h5(h5_filename, data, label, data_dtype='uint8', label_dtype='uint8'):
        h5_fout = h5py.File(h5_filename, 'w') # add 'w' to write
        h5_fout.create_dataset(
                'data', data=data,
                compression='gzip', compression_opts=4,
                dtype=data_dtype)
        h5_fout.create_dataset(
                'label', data=label,
                compression='gzip', compression_opts=1,
                dtype=label_dtype)
        h5_fout.close()

    # Set paths
    data_label_files = [os.path.join(data_dir, file) for file in os.listdir(data_dir) if file.endswith('.npy')]
    output_dir = output_h5
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    output_filename_prefix = os.path.join(output_dir, 'ply_data_all')
    output_room_filelist = os.path.join(output_dir, 'room_filelist.txt')
    output_all_file = os.path.join(output_dir, 'all_files.txt')
    fout_room = open(output_room_filelist, 'w')
    all_file = open(output_all_file, 'w')

    # Create blocks
    # Adopted from: https://github.com/charlesq34/pointnet
    # Generate blocks for training data
    # Blocks are stored in h5
    # --------------------------------------
    # ----- BATCH WRITE TO HDF5 -----
    # --------------------------------------
    batch_data_dim = [H5_BATCH_SIZE] + data_dim
    batch_label_dim = [H5_BATCH_SIZE] + label_dim
    h5_batch_data = np.zeros(batch_data_dim, dtype = np.float32)
    h5_batch_label = np.zeros(batch_label_dim, dtype = np.uint8)
    buffer_size = 0  # state: record how many samples are currently in buffer
    h5_index = 0 # state: the next h5 file to save

    def check_block(data_label, block_size, stride):
        data_label = np.load(data_label_filename)
        data = data_label[:, 0:-1]
        limit = np.amax(data, 0)[0:3]
        num_block_x = int(np.ceil((limit[0] - block_size) / stride)) + 1
        num_block_y = int(np.ceil((limit[1] - block_size) / stride)) + 1
        print('num_block_x', num_block_x)
        print('num_block_y', num_block_y)

        if num_block_x <= 0 or num_block_y <= 0:
            return False  # Area cannot be split into blocks

        return True

    def insert_batch(data, label, last_batch=False):
        global h5_batch_data, h5_batch_label
        global buffer_size, h5_index
        data_size = data.shape[0]
        # If there is enough space, just insert
        if buffer_size + data_size <= h5_batch_data.shape[0]:
            h5_batch_data[buffer_size:buffer_size+data_size, ...] = data
            h5_batch_label[buffer_size:buffer_size+data_size] = label
            buffer_size += data_size
        else: # not enough space
            capacity = h5_batch_data.shape[0] - buffer_size
            assert(capacity>=0)
            if capacity > 0:
                h5_batch_data[buffer_size:buffer_size+capacity, ...] = data[0:capacity, ...]
                h5_batch_label[buffer_size:buffer_size+capacity, ...] = label[0:capacity, ...]
            # Save batch data and label to h5 file, reset buffer_size
            h5_filename =  output_filename_prefix + '_' + str(h5_index) + '.h5'
            save_h5(h5_filename, h5_batch_data, h5_batch_label, data_dtype, label_dtype)
            print('Stored {0} with size {1}'.format(h5_filename, h5_batch_data.shape[0]))
            h5_index += 1
            buffer_size = 0
            # recursive call
            insert_batch(data[capacity:, ...], label[capacity:, ...], last_batch)
        if last_batch and buffer_size > 0:
            h5_filename =  output_filename_prefix + '_' + str(h5_index) + '.h5'
            save_h5(h5_filename, h5_batch_data[0:buffer_size, ...], h5_batch_label[0:buffer_size, ...], data_dtype, label_dtype)
            print('Stored {0} with size {1}'.format(h5_filename, buffer_size))
            h5_index += 1
            buffer_size = 0
        return

    blockability_dict = {}
    for i, data_label_filename in enumerate(data_label_files):
        print(data_label_filename)
        is_blockable = check_block(data_label_filename, block_size=block_size, stride=stride)
        blockability_dict[data_label_filename] = is_blockable

    for area, is_blockable in blockability_dict.items():
        if is_blockable:
            print(f"{area} can be split into blocks.")
        else:
            print(f"{area} cannot be split into blocks.")

    sample_cnt = 0
    for i, data_label_filename in enumerate(data_label_files):
        # print(data_label_filename)
        data, label = room2blocks_wrapper_normalized(data_label_filename, NUM_POINT, block_size=block_size, stride=stride,
                                                            random_sample=random_sample, sample_num=sample_num)
        print('{0}, {1}'.format(data.shape, label.shape))
        for _ in range(data.shape[0]):
            fout_room.write(os.path.basename(data_label_filename)[0:-4]+'\n')

        sample_cnt += data.shape[0]
        insert_batch(data, label, i == len(data_label_files)-1)

    fout_room.close()
    print("Total samples: {0}".format(sample_cnt))

    for i in range(h5_index):
        all_file.write(os.path.join(output_h5[5:], 'ply_data_all_') + str(i) +'.h5\n') # Check output name
    all_file.close()

if __name__ == "__main__":
    main()