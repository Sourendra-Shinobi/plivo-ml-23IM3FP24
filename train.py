import argparse
import csv
import os
import joblib
import numpy as np
import librosa
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline

from features import load_wav, speech_before, frame_energy_db, f0_contour

def extract_features(x, sr, pause_start):
    """Advanced feature extraction using Librosa for spectral traits."""
    seg = speech_before(x, sr, pause_start, window_s=1.5)
    
    # Fallback for extremely short segments to prevent crashes
    if len(seg) < 512:
        return np.zeros(26, dtype=np.float32)
        
    # --- 1. Base Features (from starter) ---
    e = frame_energy_db(seg, sr)
    f0 = f0_contour(seg, sr)
    voiced = f0[f0 > 0]
    
    e_mean = e.mean() if len(e) > 0 else 0
    f0_mean = voiced.mean() if len(voiced) > 0 else 0
    end_e = e[-15:].mean() if len(e) >= 15 else e_mean
    end_f0 = voiced[-15:].mean() if len(voiced) >= 15 else f0_mean
    
    e_drop = end_e - e_mean 
    f0_ratio = (end_f0 / (f0_mean + 1e-5)) 
    end_voice_density = np.sum(f0[-15:] > 0) / 15.0 if len(f0) >= 15 else 0.0

    # --- 2. Advanced Spectral Features (LIBROSA) ---
    seg_float = seg.astype(np.float32)
    
    # Zero Crossing Rate (detects breaths/fricatives at turn ends)
    zcr = librosa.feature.zero_crossing_rate(y=seg_float)[0]
    zcr_mean = zcr.mean()
    zcr_end = zcr[-5:].mean() if len(zcr) >= 5 else zcr_mean
    
    # Spectral Centroid (brightness of sound, drops when trailing off)
    cent = librosa.feature.spectral_centroid(y=seg_float, sr=sr)[0]
    cent_mean = cent.mean()
    cent_end = cent[-5:].mean() if len(cent) >= 5 else cent_mean
    
    # MFCCs (Mel-frequency cepstral coefficients - captures vocal tract shape)
    # We take the mean and std of the first 5 MFCCs
    mfcc = librosa.feature.mfcc(y=seg_float, sr=sr, n_mfcc=5)
    mfcc_means = mfcc.mean(axis=1) # 5 features
    mfcc_stds = mfcc.std(axis=1)   # 5 features

    # Compile all features into a fixed-length array (26 features)
    feats = [
        e_mean, e.std() if len(e) > 0 else 0, 
        f0_mean, voiced.std() if len(voiced) > 0 else 0,
        end_e, end_f0, e_drop, f0_ratio, end_voice_density,
        len(voiced) / max(1, len(f0)), len(seg) / sr,
        zcr_mean, zcr_end, cent_mean, cent_end
    ]
    feats.extend(mfcc_means)
    feats.extend(mfcc_stds)
    
    return np.array(feats, dtype=np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out", default="predictions.csv")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(os.path.join(args.data_dir, "labels.csv"))))
    cache = {}
    X, y, groups, keys = [], [], [], []
    for r in rows:
        path = os.path.join(args.data_dir, r["audio_file"])
        if path not in cache:
            cache[path] = load_wav(path)
        x, sr = cache[path]
        X.append(extract_features(x, sr, float(r["pause_start"])))
        y.append(1 if r["label"] == "eot" else 0)
        groups.append(r["turn_id"])
        keys.append((r["turn_id"], r["pause_index"]))
    
    X, y = np.array(X), np.array(y)

    # Standout Optimization: Stronger RF to handle 26 features without overfitting
    clf = make_pipeline(
        StandardScaler(),
        RandomForestClassifier(n_estimators=300, max_depth=12, class_weight="balanced_subsample", random_state=42)
    )
    
    clf.fit(X, y)
    joblib.dump(clf, "model.pkl")

    p = clf.predict_proba(X)[:, 1]
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["turn_id", "pause_index", "p_eot"])
        for (tid, pi), pi_p in zip(keys, p):
            w.writerow([tid, pi, f"{pi_p:.4f}"])
            
    print(f"wrote {len(keys)} predictions -> {args.out}")
    print("Advanced Librosa Model saved successfully!")

if __name__ == "__main__":
    main()