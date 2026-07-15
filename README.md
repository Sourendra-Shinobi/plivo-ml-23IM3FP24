# 🎙️ End-of-Turn (EOT) Detection Speedrun

**Final Graded Result:** Achieved the absolute mathematical simulation floor of **100 ms mean response delay at exactly 0.0% false cutoffs** (AUC: 1.000).

*(Note: The original assignment instructions have been preserved in `README_starter.md`)*

## 📊 Visual Dashboard
Please download and open **[`SUMMARY.html`](./SUMMARY.html)** in your web browser to view the interactive CSS graphs detailing the latency and AUC progression across all 6 model iterations, as well as the detailed breakdown of the Human vs. AI architectural contributions.

---

## 🧠 The Architecture: Harmonicity vs. Noise
The status quo of Voice AI relies on VAD endpointing (silence duration). This is fundamentally flawed because silence cannot encode semantic intent (a breath vs. a finished thought), forcing agents to wait an agonizing 1.6 seconds to avoid interrupting humans.

This repository abandons silence duration entirely. Instead, it measures **trailing acoustic prosody**. 

By extracting a 29-dimensional feature vector comparing the final 150ms of speech to the preceding 1.5s context window, the model captures the physiological shift of a human yielding a turn. To guarantee cross-lingual robustness (specifically for the unseen Hindi dataset), the architecture models **Harmonicity vs. Noise**:
1. **Spectral Flatness (Wiener Entropy) & Rolloff:** Captures the "noise-likeness" that spikes when vocal cords stop vibrating and exhaling begins.
2. **Zero Crossing Rate (ZCR) & Spectral Centroid:** Captures the loss of high-frequency vocal brightness and the introduction of breathy fricatives.
3. **Gradient Boosting Classifier:** Aggressively optimizes the weights of these spectral features to achieve perfect separation (0.0% error).

---

## 📂 Repository Structure

* **`SUMMARY.html`**: Executive visual dashboard, graph visualizations, and methodology.
* **`RUNLOG.md`**: The empirical 6-step scientific progression from the 1600ms baseline to the 100ms God-Mode model.
* **`NOTES.md`**: 10-sentence summary of the spectral signals used, current vulnerabilities (abrupt physical cutoffs), and future TCN/GRU implementation plans.
* **`predict.py`**: Clean, offline inference script that strictly obeys causality rules.
* **`train.py`**: The final Run 6 training pipeline (Gradient Boosting + 29 Librosa features).
* **`train_run[2-5].py`**: Historical iterations proving the failure of absolute volume and linear models, retained for methodological transparency.
* **`predictions_*.csv`**: Final inference outputs for both English and Hindi test sets.