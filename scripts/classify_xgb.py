import time
import pickle
import argparse
import os
import numpy as np
import xgboost as xgb

try:
    import laspy
except ImportError:
    laspy = None

feat_to_use = [0, 1, 2, 3, 4, 5]     # Indices of the features to use (X, Y, Z, R, G, B)

def read_model(filepath):
    ''' Read the XGBoost model from a .pkl file '''
    return pickle.load(open(filepath, 'rb'))

def read_data(las_file):
    lasfile = laspy.read(las_file)
    x = lasfile.x
    y = lasfile.y
    z = lasfile.z
    r = lasfile.red
    g = lasfile.green
    b = lasfile.blue

    X_train = np.column_stack((x, y, z, r, g, b))
    return X_train

def write_classification(X, Y, filename):
    header = laspy.LasHeader(point_format=2, version="1.2")
    las = laspy.LasData(header)
    las.x = X[:, 0]
    las.y = X[:, 1]
    las.z = X[:, 2]
    las.red = X[:, 3]
    las.green = X[:, 4]
    las.blue = X[:, 5]
    
    # Mapping otomatis (0->2, 1->5, 2->6)
    # Sesuaikan jika XGBoost Anda sudah langsung memprediksi 2, 5, 6
    map_array = np.array([2, 5, 6]) 
    mapped_class = map_array[Y.astype(int)]
    las.classification = mapped_class.astype(np.uint8)
    
    las.write(filename)

def main():
    parser = argparse.ArgumentParser(description='Classify a point cloud with a XGBoost model.')
    # Ubah menjadi positional flags sesuai format dari main.py
    parser.add_argument('--model', required=True, help='Path to .pkl file containing the trained model.')
    parser.add_argument('--point_cloud', required=True, help='Path to .las file containing the point cloud to classify.')
    parser.add_argument('--output_dir', required=True, help='Output directory.')
    # Parameter dummy agar GUI tidak error saat mengirim parameter ini
    parser.add_argument('--batch_size', type=int, default=16, help='Ignored for XGBoost')
    args = parser.parse_args()

    start = time.time() 
    print('Loading data ...', flush=True)
    model = read_model(args.model)
    
    # ---------------------------------------------------------
    # PAKSA XGBOOST MENGGUNAKAN GPU UNTUK INFERENCE
    # ---------------------------------------------------------
    try:
        if hasattr(model, 'set_param'):
            model.set_param({'device': 'cuda'}) # Untuk XGBoost terbaru >= 2.0
        else:
            model.set_params(device='cuda')
    except:
        try:
            if hasattr(model, 'set_param'):
                model.set_param({'predictor': 'gpu_predictor'}) # Untuk XGBoost lama
            else:
                model.set_params(predictor='gpu_predictor')
        except:
            pass

    X = read_data(args.point_cloud)
    
    print('Classifying the dataset ...', flush=True)
    
    # Pengecekan tipe model agar tidak crash jika di-export dengan format berbeda
    if isinstance(model, xgb.Booster):
        dtest = xgb.DMatrix(data=X[:, feat_to_use])
        Y_pred_raw = model.predict(dtest)
        
        # Jika model mengembalikan probabilitas, ambil nilai tertingginya
        if len(Y_pred_raw.shape) > 1 and Y_pred_raw.shape[1] > 1:
            Y_pred = np.argmax(Y_pred_raw, axis=1)
        else:
            Y_pred = Y_pred_raw
    else:
        # Jika model merupakan scikit-learn API wrapper (XGBClassifier)
        Y_pred = model.predict(X[:, feat_to_use])

    print('Saving ...', flush=True)
    
    # ---------------------------------------------------------
    # PEMBUATAN NAMA FILE OTOMATIS (_classified.las)
    # ---------------------------------------------------------
    filename = os.path.basename(args.point_cloud)
    name, ext = os.path.splitext(filename)
    output_name = os.path.join(args.output_dir, f"{name}_classified.las")
    
    write_classification(X, Y_pred, output_name)
    end = time.time()
    
    # Sinyal Progress untuk GUI
    print("PROGRESS:1/1", flush=True)
    print(f"Data classified in: {end - start:.2f} seconds", flush=True)

if __name__== '__main__':
    main()