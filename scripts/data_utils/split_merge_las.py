import laspy
import numpy as np
import argparse

def split_indices(data, num_blocks_x, num_blocks_y, block_size):
    """Like split_array, but returns the row indices of the points in each block."""
    min_x = data[:, 0].min()
    min_y = data[:, 1].min()
    bx = np.minimum(((data[:, 0] - min_x) // block_size).astype(np.int64), num_blocks_x - 1)
    by = np.minimum(((data[:, 1] - min_y) // block_size).astype(np.int64), num_blocks_y - 1)
    block_id = bx * num_blocks_y + by
    order = np.argsort(block_id, kind='stable')
    bounds = np.searchsorted(block_id[order], np.arange(num_blocks_x * num_blocks_y + 1))
    return [order[bounds[k]:bounds[k + 1]] for k in range(num_blocks_x * num_blocks_y)]

def read_las(las_files):
    inFile = laspy.read(las_files) # read a las file
    x = inFile.x
    y = inFile.y
    z = inFile.z
    r = inFile.red
    g = inFile.green
    b = inFile.blue
    data = np.column_stack((x, y, z, r, g, b))
    return data

def calculate_block_size(data, block_size):
    """
    Calculate the number of blocks in each dimension based on a fixed block size.

    Args:
    - data: numpy array containing point cloud data
    - block_size: size of each block in meters
    
    Returns:
    - Number of blocks in each dimension
    """
    # Calculate the maximum extent in x and y directions
    max_extent_x = np.max(data[:, 0])
    min_extent_x = np.min(data[:, 0])
    max_extent_y = np.max(data[:, 1])
    min_extent_y = np.min(data[:, 1])
    
    # Calculate the number of blocks in x and y directions
    num_blocks_x = int((max_extent_x - min_extent_x) / block_size) + 1
    num_blocks_y = int((max_extent_y - min_extent_y) / block_size) + 1
    
    return num_blocks_x, num_blocks_y

def split_array(data, num_blocks_x, num_blocks_y, block_size):
    """
    Split the numpy array into blocks based on the number of blocks in each dimension.

    Args:
    - data: numpy array containing point cloud data
    - num_blocks_x: number of blocks in the x direction
    - num_blocks_y: number of blocks in the y direction
    - block_size: size of each block in meters
    
    Returns:
    - List of numpy arrays, each representing a block
    """
    blocks = []
    # Calculate the minimum extent of x and y coordinates
    min_extent_x = np.min(data[:, 0])
    min_extent_y = np.min(data[:, 1])
    for i in range(num_blocks_x):
        for j in range(num_blocks_y):
            # Calculate the start and end indices for the current block in x and y directions
            start_x = min_extent_x + i * block_size
            end_x = start_x + block_size
            start_y = min_extent_y + j * block_size
            end_y = start_y + block_size
            
            # Filter points within the current block
            block_points = data[(data[:, 0] >= start_x) & (data[:, 0] < end_x) &
                                (data[:, 1] >= start_y) & (data[:, 1] < end_y)]
            blocks.append(block_points)
    return blocks


def merge_blocks(blocks):
    """
    Merge the blocks back into a single numpy array.
    
    Args:
    - blocks: List of numpy arrays, each representing a block
    
    Returns:
    - Merged numpy array containing all points
    """
    return np.vstack(blocks)