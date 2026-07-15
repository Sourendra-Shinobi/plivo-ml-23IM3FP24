import argparse
import csv
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from features import load_wav, speech_before, frame_energy_db, f0_contour

def extract_features(x, sr, pause_start):
    """Basic features: just checking the very last few frames of energy and pitch."""
    seg = speech_before(x, sr, pause_start, window_s=1.5)
    if len(seg) < sr // 10: return np.zeros(3, dtype=np.float32)
    e = frame_energy_db(seg, sr)
    f0 = f0_contour(seg, sr)
    voiced = f0[f0 > 0]
    return np.array([
        e[-5:].mean() if len(e)>=5 else 0,
        voiced[-3:].mean() if len(voiced)>=3 else 0,
        len(seg)/sr
    ], dtype=np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out", default="predictions_run2.csv")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(os.path.join(args.data_dir, "labels.csv"))))
    cache = {}
    X, y, keys = [], [], []
    for r in rows:
        path = os.path.join(args.data_dir, r["audio_file"])
        if path not in cache: cache[path] = load_wav(path)
        x, sr = cache[path]
        X.append(extract_features(x, sr, float(r["pause_start"])))
        y.append(1 if r["label"] == "eot" else 0)
        keys.append((r["turn_id"], r["pause_index"]))
    
    X, y = np.array(X), np.array(y)
    
    clf = LogisticRegression(class_weight="balanced", max_iter=1000)
    clf.fit(X, y)
    
    p = clf.predict_proba(X)[:, 1]
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["turn_id", "pause_index", "p_eot"])
        for (tid, pi), pi_p in zip(keys, p): w.writerow([tid, pi, f"{pi_p:.4f}"])

if __name__ == "__main__": 
    main()