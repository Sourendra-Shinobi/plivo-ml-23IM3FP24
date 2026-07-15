import argparse
import csv
import os
import joblib
import numpy as np
from features import load_wav
from train import extract_features

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out", default="predictions.csv")
    args = ap.parse_args()

    # Load the trained model saved by train.py
    model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Run train.py first!")
    clf = joblib.load(model_path)

    rows = list(csv.DictReader(open(os.path.join(args.data_dir, "labels.csv"))))
    cache = {}
    X, keys = [], []
    for r in rows:
        path = os.path.join(args.data_dir, r["audio_file"])
        if path not in cache:
            cache[path] = load_wav(path)
        x, sr = cache[path]
        X.append(extract_features(x, sr, float(r["pause_start"])))
        keys.append((r["turn_id"], r["pause_index"]))
        
    X = np.array(X)
    p = clf.predict_proba(X)[:, 1]
    
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["turn_id", "pause_index", "p_eot"])
        for (tid, pi), pi_p in zip(keys, p):
            w.writerow([tid, pi, f"{pi_p:.4f}"])
            
    print(f"wrote {len(keys)} predictions -> {args.out}")

if __name__ == "__main__":
    main()