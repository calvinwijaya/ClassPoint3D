import os
import glob
import time
import pickle
import argparse
import itertools
import numpy as np
import laspy
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

global feat_to_use, class_index
feat_to_use = [0, 1, 2, 3, 4, 5]
class_index = -1
debug = True
x_column_averages = [0.0, 0.0]


def las_to_features(lasfile, idx=None):
    """Build the feature matrix (x, y, z, r, g, b), optionally only for selected point indices."""
    if idx is None:
        idx = slice(None)
    return np.column_stack((
        np.asarray(lasfile.x)[idx],
        np.asarray(lasfile.y)[idx],
        np.asarray(lasfile.z)[idx],
        np.asarray(lasfile.red)[idx],
        np.asarray(lasfile.green)[idx],
        np.asarray(lasfile.blue)[idx],
    ))


def read_data(las_file):
    """Read one LAS/LAZ file fully: features and raw (unshifted) class codes."""
    lasfile = laspy.read(las_file)
    X = las_to_features(lasfile)
    Y = np.array(lasfile.classification)
    return X, Y


def sample_indices(Y, max_points, min_per_class, rng):
    """
    Stratified random sample of point indices.
    Each class keeps roughly its original proportion, but rare classes keep at
    least `min_per_class` points (or all of them if they have fewer).
    """
    n = len(Y)
    if max_points is None or max_points <= 0 or n <= max_points:
        return np.arange(n)

    classes, counts = np.unique(Y, return_counts=True)
    picked = []
    for c, cnt in zip(classes, counts):
        quota = int(round(max_points * cnt / n))
        quota = max(quota, min(cnt, min_per_class))
        quota = min(quota, cnt)
        cidx = np.flatnonzero(Y == c)
        picked.append(rng.choice(cidx, quota, replace=False))
    return np.sort(np.concatenate(picked))


def list_las_files(folder, recursive=False):
    """Return a sorted list of .las/.laz files in a folder."""
    patterns = ['*.las', '*.LAS', '*.laz', '*.LAZ']
    files = []
    for p in patterns:
        if recursive:
            files.extend(glob.glob(os.path.join(folder, '**', p), recursive=True))
        else:
            files.extend(glob.glob(os.path.join(folder, p)))
    return sorted(set(os.path.normcase(os.path.abspath(f)) for f in files))


def read_training_data(train_file=None, train_dir=None, recursive=False, exclude=None,
                       max_points_per_file=2_000_000, min_per_class=20_000, seed=0,
                       keep_classes=None):
    """Load (sampled) training data from a single file or every LAS/LAZ file in a folder."""
    if train_file is not None:
        files = [os.path.normcase(os.path.abspath(train_file))]
    else:
        if not os.path.isdir(train_dir):
            raise NotADirectoryError('Training folder not found: {}'.format(train_dir))
        files = list_las_files(train_dir, recursive)

    # Never train on the test file
    if exclude is not None:
        excl = os.path.normcase(os.path.abspath(exclude))
        if excl in files:
            print('\tSkipping test file found in training folder: {}'.format(os.path.basename(exclude)))
            files = [f for f in files if f != excl]

    if len(files) == 0:
        raise FileNotFoundError('No training .las/.laz files left to use.')

    rng = np.random.default_rng(seed)
    X_list, Y_list = [], []
    for f in files:
        lasfile = laspy.read(f)
        Y_full = np.array(lasfile.classification)

        # Keep only points of the wanted classes, then sample among them
        if keep_classes is not None:
            keep_idx = np.flatnonzero(np.isin(Y_full, keep_classes))
        else:
            keep_idx = np.arange(len(Y_full))
        sub = sample_indices(Y_full[keep_idx], max_points_per_file, min_per_class, rng)
        idx = keep_idx[sub]

        X_list.append(las_to_features(lasfile, idx))
        Y_list.append(Y_full[idx])
        print('\t- {} ({} points, {} in kept classes, sampled {}, classes: {})'.format(
            os.path.basename(f), len(Y_full), len(keep_idx), len(idx), np.unique(Y_full).tolist()))
        del lasfile, Y_full  # free memory before the next file

    return np.vstack(X_list), np.concatenate(Y_list)


def build_label_mapping(Y_train_raw):
    """Map original class codes (e.g. 2, 5, 6) to contiguous 0..K-1 indices."""
    classes = np.unique(Y_train_raw)
    to_index = {int(c): i for i, c in enumerate(classes)}
    return classes, to_index


def encode_labels(Y_raw, classes, name='data'):
    """Vectorized encoding with the training class list. Unknown classes get -1."""
    pos = np.clip(np.searchsorted(classes, Y_raw), 0, len(classes) - 1)
    known = classes[pos] == Y_raw
    Y = np.where(known, pos, -1).astype(np.int64)
    n_unknown = int(np.sum(~known))
    if n_unknown > 0:
        unknown = np.unique(Y_raw[~known]).tolist()
        print('\tWarning: {} points in {} have classes not seen in training: {}'.format(n_unknown, name, unknown))
    return Y


def write_classification(X, Y, filename):
    # Keep .las / .laz if given, otherwise default to .las
    if not filename.lower().endswith(('.las', '.laz')):
        filename = filename + '.las'

    header = laspy.LasHeader(point_format=2, version="1.2")
    header.offsets = np.floor(X[:, :3].min(axis=0))
    header.scales = np.array([0.001, 0.001, 0.001])
    las = laspy.LasData(header)
    las.x = X[:, 0]
    las.y = X[:, 1]
    las.z = X[:, 2]
    las.red = X[:, 3]
    las.green = X[:, 4]
    las.blue = X[:, 5]

    las.classification = Y.astype(np.uint8)
    las.write(filename)
    print('Saved prediction to: {}'.format(filename))


def train_model(X_train, Y_train, n_estimators, max_depth, num_class):
    # QuantileDMatrix uses much less memory than DMatrix with tree_method='hist'
    dtrain = xgb.QuantileDMatrix(data=X_train[:, feat_to_use].astype(np.float32), label=Y_train)
    params = {
        'objective': 'multi:softmax',   # For classification
        'num_class': num_class,         # Number of classes
        'tree_method': 'hist',
        'device': 'cuda',               # Use GPU for training
        'eval_metric': 'mlogloss',
        'random_state': 0,
        'seed': 0
    }
    if max_depth is not None:
        params['max_depth'] = max_depth
    model = xgb.train(params, dtrain, num_boost_round=n_estimators)
    del dtrain
    return model


def predict_model(model, X_test, batch_size=5_000_000):
    """Predict in batches to keep memory usage low on large test files."""
    preds = np.empty(len(X_test), dtype=np.int64)
    for start in range(0, len(X_test), batch_size):
        end = min(start + batch_size, len(X_test))
        dtest = xgb.DMatrix(data=X_test[start:end, feat_to_use].astype(np.float32))
        preds[start:end] = model.predict(dtest).astype(np.int64)
        del dtest
    return preds


def save_model(model, filename):
    with open(filename, 'wb') as out:
        pickle.dump(model, out, pickle.HIGHEST_PROTOCOL)


def main():
    parser = argparse.ArgumentParser(description='Train the XGBoost model.')
    train_group = parser.add_mutually_exclusive_group(required=True)
    train_group.add_argument('--train_file', help='Path to a single training LAS/LAZ file')
    train_group.add_argument('--train_dir', help='Path to a folder containing training LAS/LAZ files')
    parser.add_argument('--recursive', action='store_true', help='Also search subfolders of --train_dir')
    parser.add_argument('--max_points_per_file', type=int, default=2_000_000,
                        help='Max training points sampled per file (0 = use all points). Default: 2,000,000')
    parser.add_argument('--min_per_class', type=int, default=20_000,
                        help='Minimum points kept per class per file, so rare classes are not lost. Default: 20,000')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for sampling')
    parser.add_argument('--keep_classes', type=int, nargs='+', default=[2, 5, 6],
                        help='Only train/evaluate on these LAS classes. Default: 2 5 6')
    parser.add_argument('--predict_all', action='store_true',
                        help='Write ALL test points to the output (each gets one of the kept classes). '
                             'By default only test points of the kept classes are written.')
    parser.add_argument('test_filepath', help='Path to the test file')
    parser.add_argument('output_name', help='Name of the predicted test file (.las or .laz)')
    args = parser.parse_args()

    print("Loading training data...")
    X_train, Y_train_raw = read_training_data(
        args.train_file, args.train_dir, args.recursive, exclude=args.test_filepath,
        max_points_per_file=args.max_points_per_file, min_per_class=args.min_per_class, seed=args.seed,
        keep_classes=args.keep_classes)

    print("Loading test data...")
    X_test_all, Y_test_all_raw = read_data(args.test_filepath)
    test_mask = np.isin(Y_test_all_raw, args.keep_classes)
    X_test = X_test_all[test_mask]
    Y_test_raw = Y_test_all_raw[test_mask]
    print('\tTest points in kept classes {}: {} of {}'.format(args.keep_classes, len(Y_test_raw), len(Y_test_all_raw)))
    if not args.predict_all:
        del X_test_all, Y_test_all_raw  # free memory, not needed

    # Fixed label mapping from the kept classes (e.g. 2 -> 0, 5 -> 1, 6 -> 2)
    classes, to_index = build_label_mapping(np.array(args.keep_classes))
    missing = [int(c) for c in classes if c not in np.unique(Y_train_raw)]
    if missing:
        raise ValueError('These kept classes have no training points: {}'.format(missing))
    num_class = len(classes)
    Y_train = encode_labels(Y_train_raw, classes, 'training')
    Y_test = encode_labels(Y_test_raw, classes, 'test')
    print('\tClass mapping (original -> index): {}'.format(to_index))
    print('\tTraining samples: {}\n\tTesting samples: {}\n\tUsing features with indices: {}'.format(
        len(Y_train), len(Y_test), feat_to_use))

    n_estimators = [50, 100, 150, 200]
    max_depths = [None]
    best_conf = {'ne': 0, 'md': 0}
    best_f1 = -1
    best_pred = None

    # Train model
    print('\nTraining the model...')
    start = time.time()
    for ne, md in list(itertools.product(n_estimators, max_depths)):
        model = train_model(X_train, Y_train, ne, md, num_class)
        Y_test_pred = predict_model(model, X_test)
        acc = accuracy_score(Y_test, Y_test_pred)
        f1 = f1_score(Y_test, Y_test_pred, average='weighted')
        if f1 > best_f1:
            best_conf['ne'] = ne
            best_conf['md'] = md
            best_f1 = f1
            best_pred = Y_test_pred

        if debug:
            print('\tne: {}, md: {} - acc: {} f1: {}'.format(ne, md, acc, f1))
    end = time.time()
    print('---> Best parameters: ne: {}, md: {}'.format(best_conf['ne'], best_conf['md']))
    print('---> Confusion matrix (best model), rows = true, cols = predicted, order = {}:\n{}'.format(
        classes.tolist(), confusion_matrix(Y_test, best_pred, labels=list(range(num_class)))))
    print('---> Training time: {} seconds'.format(end - start))
    ''' ******************************************************************************************** '''

    model = train_model(X_train, Y_train, best_conf['ne'], best_conf['md'], num_class)
    X_out = X_test_all if args.predict_all else X_test
    Y_out_pred = predict_model(model, X_out)

    # Convert predicted indices back to the original LAS class codes before writing
    Y_out_codes = classes[Y_out_pred]
    write_classification(X_out, Y_out_codes, args.output_name)

    save_model(model, 'ne{}_md{}.pkl'.format(best_conf['ne'], best_conf['md']))
    with open('ne{}_md{}_classes.pkl'.format(best_conf['ne'], best_conf['md']), 'wb') as out:
        pickle.dump(classes, out, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()
