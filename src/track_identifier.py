import librosa
import numpy as np
from statistics import mode
import joblib
import warnings
import shutil
from xgboost import XGBClassifier
from sklearn import *

scaler = joblib.load("./pretrained_models/identification/min_max_scaler.save")
model = joblib.load("./pretrained_models/identification/xgb_5s_model.save")

min_distance = 15
song_err = []


async def genre_predict(file_path: str):
    y, sr = librosa.load(file_path)
    labels = ["EDM", "Pop", "RnB", "Rock", "Trap"]
    predictions = []

    for i in range(2):
        # Calculate a non-negative start_time
        start_time = max(0, np.random.uniform(0.2 * len(y), 0.8 * len(y) - 30 * sr))

        while i > 0 and abs(start_time - prev_start_time) < min_distance * sr:
            start_time = max(0, np.random.uniform(0.2 * len(y), 0.8 * len(y) - 30 * sr))

        start_time = int(start_time)

        if start_time < 0:
            print(f"error: {file_path}")
            song_err.append(file_path)

        # split 5s
        for j in range(6):
            part_start = start_time + j * 5 * sr
            part_end = part_start + 5 * sr
            y_split = y[part_start:part_end]
            x_features = get_feature(y_split, sr)
            x_norm = scaler.transform([x_features])
            y_pred = model.predict(x_norm)
            predictions.append(labels[y_pred[0]])

        prev_start_time = start_time

    result = mode(predictions)

    return result


def get_feature(y, sr):
    # length
    length = y.shape[0]

    # chroma_stft
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)

    # rms
    rms = librosa.feature.rms(y=y)

    # spectral_centroid
    spectral_centroid = librosa.feature.spectral_centroid(y=y)

    # spectral_bandwidth
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y)

    # rolloff
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y)

    # zero_crossing_rate
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y=y)

    # harmony and perceptr
    harmony, perceptr = librosa.effects.hpss(y=y)

    # tempo
    tempo, _ = librosa.beat.beat_track(y=y)

    # mfcc
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfccs_mean = np.mean(mfccs, axis=1)
    mfccs_var = np.var(mfccs, axis=1)

    # concat all features
    features = [
        length,
        chroma_stft.mean(),
        chroma_stft.var(),
        rms.mean(),
        rms.var(),
        spectral_centroid.mean(),
        spectral_centroid.var(),
        spectral_bandwidth.mean(),
        spectral_bandwidth.var(),
        spectral_rolloff.mean(),
        spectral_rolloff.var(),
        zero_crossing_rate.mean(),
        zero_crossing_rate.var(),
        harmony.mean(),
        harmony.var(),
        perceptr.mean(),
        perceptr.var(),
        tempo,
        *mfccs_mean,
    ]

    return features
