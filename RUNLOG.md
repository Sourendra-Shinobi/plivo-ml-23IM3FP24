# Run Log: End-of-Turn Detection

**Run 1 (Status Quo Baseline)**
*   **Score:** 1600 ms delay | AUC: 0.514
*   **What changed:** Ran the provided `baseline.py`.
*   **Why:** To establish the baseline of purely relying on a silence-duration timer (VAD endpointing). 

**Run 2 (Absolute Prosody)**
*   **Score:** 1190 ms delay | AUC: 0.599
*   **What changed:** Replaced baseline with Logistic Regression using absolute mean pitch and energy from the final 150ms of speech.
*   **Why:** Hypothesized that humans drop pitch and volume when yielding a turn, but needed to test if absolute thresholds were sufficient indicators.

**Run 3 (Relative Prosodic Context)**
*   **Score:** 1322 ms delay | AUC: 0.605
*   **What changed:** Added relative slope features (End value minus Context Mean value) to the Logistic Regression model.
*   **Why:** Discovered that absolute volume fails (a loud speaker pausing is still louder than a quiet speaker stopping), requiring relative context to measure the drop.

**Run 4 (Non-Linear Architecture)**
*   **Score:** 235 ms delay | AUC: 0.990
*   **What changed:** Swapped the Logistic Regression for a tree-based `RandomForestClassifier`.
*   **Why:** Linear models cannot draw complex boundaries around acoustic slopes; tree-based models naturally map these non-linear intersections.

**Run 5 (Timbral Generalization)**
*   **Score:** 100 ms delay | 1.0% Cutoffs | AUC: 1.000
*   **What changed:** Added `librosa` spectral features (ZCR, MFCCs, Spectral Centroid) to capture vocal tract shape and breathiness.
*   **Why:** Intonation rules (pitch/volume) overfit to English; timbral shifts (loss of brightness, increase in breath) generalize better to unseen languages like Hindi.

**Run 6 (Harmonicity vs Noise)**
*   **Score:** 100 ms delay | 0.0% Cutoffs | AUC: 1.000
*   **What changed:** Upgraded to `GradientBoostingClassifier` and added Spectral Flatness (Wiener Entropy) and Spectral Rolloff.
*   **Why:** To perfectly isolate the exact moment vocal cords stop vibrating and breath (noise) begins, squeezing out the final 1% of errors by boosting critical spectral feature weights.