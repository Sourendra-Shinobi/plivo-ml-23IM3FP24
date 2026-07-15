1. The model relies on the trailing prosody of the 150ms immediately preceding a pause, contrasted against the 1.5s context window.
2. The core signals used are pitch (F0) slopes, energy fading ratios, Zero Crossing Rate (ZCR), and Spectral Centroids.
3. These acoustic features successfully differentiate a turn-yield (characterized by dropping pitch, loss of high-frequency brightness, and increased breathiness) from a mid-sentence hold (characterized by sustained pitch and energy).
4. The model still fails on abrupt conversational cutoffs, such as a user suddenly stopping due to background noise or being interrupted, which breaks the natural prosodic curve.
5. It may also struggle with highly expressive speech where a user uses a dramatic pitch drop mid-sentence for rhetorical effect.
6. With one more day, I would abandon manual feature aggregation (means and slopes) entirely.
7. Instead, I would build a recurrent neural network (GRU) or Temporal Convolutional Network (TCN).
8. Feeding the raw frame-by-frame pitch, energy, and MFCC time-series directly into a sequential model would allow it to learn the exact temporal curvature of turn-yielding.