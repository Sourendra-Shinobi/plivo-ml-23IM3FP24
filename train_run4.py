import argparse, csv, os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from features import load_wav, speech_before, frame_energy_db, f0_contour

def extract_features(x, sr, pause_start):
    seg = speech_before(x, sr, pause_start, window_s=1.5)
    if len(seg) < sr // 10: return np.zeros(6, dtype=np.float32)
    e = frame_energy_db(seg, sr)
    f0 = f0_contour(seg, sr)
    voiced = f0[f0 > 0]
    e_mean = e.mean() if len(e) > 0 else 0
    f0_mean = voiced.mean() if len(voiced) > 0 else 0
    end_e = e[-15:].mean() if len(e) >= 15 else e_mean
    end_f0 = voiced[-15:].mean() if len(voiced) >= 15 else f0_mean
    e_drop = end_e - e_mean
    f0_drop = end_f0 - f0_mean
    return np.array([e_mean, f0_mean, end_e, end_f0, e_drop, f0_drop], dtype=np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out", default="predictions_run4.csv")
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
    
    # Swapped Logistic Regression for Random Forest
    clf = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=100, max_depth=6, class_weight="balanced", random_state=42))
    clf.fit(X, y)
    
    p = clf.predict_proba(X)[:, 1]
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["turn_id", "pause_index", "p_eot"])
        for (tid, pi), pi_p in zip(keys, p): w.writerow([tid, pi, f"{pi_p:.4f}"])

if __name__ == "__main__": main()