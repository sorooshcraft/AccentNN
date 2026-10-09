# AccentNN — Speech Accent Recognition Pipeline

An end-to-end audio processing and deep learning pipeline built with TensorFlow and `pydub` to classify spoken English accents from raw audio waveforms.

Trained on the [Speech Accent Archive](https://www.kaggle.com/datasets/rtatman/speech-accent-archive) dataset across 5 top accent classes (*Arabic, Chinese, Spanish, UK, USA*), reaching **80.3% test accuracy**.

---

## Technical Overview

### 1. Audio Preprocessing & Normalization
* **Volume Normalization:** Audio files were filtered for quality and normalized to a uniform `-20.0 dBFS` using `pydub.AudioSegment` to minimize volume bias across recording environments.
* **Format & Sampling:** Standardized audio segments to 30-second mono WAV files at 44.1 kHz (1,323,000 samples per clip).

### 2. Feature Extraction (TensorFlow Graph)
* **STFT:** Short-Time Fourier Transform computed with `frame_length=400`, `frame_step=160`, and `fft_length=512`.
* **Mel Filterbanks & MFCCs:** Converted linear spectrograms into Mel-scale representations via `tf.signal.linear_to_mel_weight_matrix`, taking the first 13 Mel-Frequency Cepstral Coefficients (MFCCs) for compact representation of acoustic timbre.

### 3. Model Architecture
* **Input Layer:** Shape `(8267, 13)` representing temporal MFCC time steps.
* **Regularization:** Injected initial `GaussianNoise(0.1)` to reduce sensitivity to ambient recording noise.
* **Recurrent Layers:** 2-layer stacked LSTM (`64` units with `return_sequences=True` $\rightarrow$ `64` units).
* **Classification Head:** `Dropout(0.25)` feeding into a 5-unit Softmax Dense output.
* **Optimization:** Adam optimizer trained with `SparseCategoricalCrossentropy` over 250 epochs.

---

## Results
* **Test Accuracy:** **80.29%** on unseen test splits (`loss: 0.6478`).
* **Inference Pipeline:** Includes an end-to-end audio preprocessing script (`predict_and_display_audio`) that pads/trims arbitrary external audio clips and outputs real-time accent predictions.
