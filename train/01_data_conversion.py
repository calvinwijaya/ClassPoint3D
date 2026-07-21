import os
import numpy as np
import laspy
import argparse

output_folder = 'data/sem_seg_data'

def parse_arguments():
    parser = argparse.ArgumentParser(description="Process LAS files and save as numpy arrays.")
    parser.add_argument("--las_file", required=False, help="Path to the LAS file.")
    parser.add_argument("--data_folder", required=False, help="Path to the folder containing LAS files.")
    return parser.parse_args()

def main():
    args = parse_arguments()   
    if args.las_file:
        las_files = [args.las_file]
    elif args.data_folder:
        if os.path.isdir(args.data_folder):
            las_files = [os.path.join(args.data_folder, f) for f in os.listdir(args.data_folder) if f.endswith('.las') or f.endswith('.LAS')]
        else:
            print("Please provide a valid directory path for --data_folder.")
            return
    else:
        print("Please provide either --las_file or --data_folder argument.")
        return

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    for idx, las_file in enumerate(las_files):
        out_filename = os.path.join(output_folder, f'Area_{idx + 1}.npy')

        lasfile = laspy.read(las_file)
        
        # 1. Ekstrak data asli
        x = np.array(lasfile.x)
        y = np.array(lasfile.y)
        z = np.array(lasfile.z)
        r = np.array(lasfile.red)
        g = np.array(lasfile.green)
        b = np.array(lasfile.blue)
        C = np.array(lasfile.classification)

        # 2. FILTERING: Buang semua titik yang BUKAN kelas 2, 5, atau 6
        valid_mask = np.isin(C, [2, 5, 6])
        
        x = x[valid_mask]
        y = y[valid_mask]
        z = z[valid_mask]
        r = r[valid_mask]
        g = g[valid_mask]
        b = b[valid_mask]
        C_filtered = C[valid_mask]

        # 3. MAPPING: Ubah (2, 5, 6) menjadi (0, 1, 2)
        mapped_C = np.zeros_like(C_filtered)
        mapped_C[C_filtered == 2] = 0
        mapped_C[C_filtered == 5] = 1
        mapped_C[C_filtered == 6] = 2

        # 4. Gabungkan array yang sudah bersih dan dimapping
        data_label = np.column_stack((x, y, z, r, g, b, mapped_C))

        # 5. Normalisasi koordinat XY
        if len(data_label) > 0:
            xy_min = np.amin(data_label, axis=0)[0:2]
            data_label[:, 0:2] -= xy_min
            
            np.save(out_filename, data_label)
            print(f"Saved {out_filename} | Total points: {len(x)}")
        else:
            print(f"Skipping {out_filename} | Tidak ada kelas 2, 5, 6 yang ditemukan.")
    
if __name__ == "__main__":
    main()