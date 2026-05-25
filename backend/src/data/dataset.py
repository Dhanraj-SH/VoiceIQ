import os
import numpy as np
import librosa
import librosa.display
from glob import glob
from src.features.extractor import extract_features
from src.data.augment import time_stretch, pitch_shift, add_noise
import soundfile as sf
import torch
from torch.utils.data import Dataset

EMOTION_MAP = {
    "angry": 0,
    "happy": 1,
    "sad": 2,
    "neutral": 3,
    "fearful": 4,
    "disgust": 5,
    "calm": 6,
    "surprised": 7
}

CREMAD_MAP = {
    "ANG": "angry",
    "DIS": "disgust",
    "FEA": "fearful",
    "HAP": "happy",
    "NEU": "neutral",
    "SAD": "sad"
}

def process_ravdess(path):
    files = glob(f"./dataset/raw/RAVDESS/*/*.wav")
    data = []
    for file in files:
        emotion = file.split("\\")[-2]
        
        if emotion not in EMOTION_MAP:
            continue

        label = EMOTION_MAP[emotion]
        data.append((file, label))
    
    return data

def process_cremad(path):
    files = glob(f"./dataset/raw/CREMAD/AudioWAV/*.wav")
    data = []
    for file in files:
        filename = file.split("\\")[-1]
        emotion_code = filename.split("_")[2]

        if emotion_code not in CREMAD_MAP:
            continue

        emotion = CREMAD_MAP[emotion_code]
        label = EMOTION_MAP[emotion]

        data.append((file, label))

    return data

def augment_and_extract(audio, sr):
    augment_features = []

    #Original
    sf.write("temp.wav", audio, sr)
    features = extract_features("temp.wav")
    augment_features.append(features)

    #Time Stretch
    stretch_audio = time_stretch(audio)
    sf.write("temp.wav", stretch_audio, sr)
    stretched_features = extract_features("temp.wav")
    augment_features.append(stretched_features)

    #Pitch Shift
    pitch_audio = pitch_shift(audio, sr)
    sf.write("temp.wav", pitch_audio, sr)
    pitch_features = extract_features("temp.wav")
    augment_features.append(pitch_features)

    #Noise
    noise_audio = add_noise(audio)
    sf.write("temp.wav", noise_audio, sr)
    noise_features = extract_features("temp.wav")
    augment_features.append(noise_features)

    return augment_features


def build_dataset(ravdess_path, cremad_path):
    ravdess_data = process_ravdess(ravdess_path)
    cremad_data = process_cremad(cremad_path)
    all_data = ravdess_data + cremad_data

    X = []
    y = []

    for idx, (file, label) in enumerate(all_data):
        if idx % 100 == 0:
            print(f"Processing {idx+1}/{len(all_data)}")
        audio, sr = librosa.load(file, sr = 22050)
        feature_list = augment_and_extract(audio, sr)
        for features in feature_list:
            X.append(features)
            y.append(label)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    print("X Shape:", X.shape)
    print("Y Shape:", y.shape)

    save_dir = os.path.join(
        os.getcwd(),
        "dataset",
        "processed"
    )

    os.makedirs(save_dir, exist_ok=True)

    np.save(os.path.join(save_dir, "X.npy"), X)

    np.save(os.path.join(save_dir, "y.npy"), y)

    print("Dataset Saved Successfully")

    if os.path.exists("temp.wav"):
        os.remove("temp.wav")

class SERDataset(Dataset):

    def __init__(self, X_path, y_path):
        self.X = np.load(X_path)
        self.y = np.load(y_path)

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):

        features = torch.tensor(self.X[idx],
                                dtype=torch.float32)

        label = torch.tensor(self.y[idx],
                             dtype=torch.long)

        return features, label