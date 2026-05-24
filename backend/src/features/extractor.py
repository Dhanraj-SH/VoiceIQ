import librosa
import numpy as np

MAX_FRAMES = 130

def pad_or_turncate(feature, max_frames = MAX_FRAMES):
    if feature.shape[1] < max_frames:
        pad_width = max_frames - feature.shape[1]
        feature = np.pad(feature,
                         pad_width=((0, 0), (0, pad_width)),
                         mode = 'constant'
        )
    else:
        feature = feature[:, :max_frames]
    return feature

def extract_features(audio_path):
    y, sr = librosa.load(audio_path, sr = 22050)

    #MFCC
    mfcc = librosa.feature.mfcc(y = y,
                                sr = sr,
                                n_mfcc = 40)
    
    mfcc_delta = librosa.feature.delta(mfcc)

    mfcc_delta2 = librosa.feature.delta(mfcc,
                                        order = 2)
    
    #MEL
    mel = librosa.feature.melspectrogram(y = y,
                                         sr = sr,
                                         n_mels = 40)
    
    mel = librosa.power_to_db(mel,
                              ref = np.max)
    
    #CHROMA
    chroma = librosa.feature.chroma_stft(y = y,
                                         sr = sr)
    
    #ZCR
    zcr = librosa.feature.spectral_contrast(y = y,
                                            sr = sr)
    
    #SPECTRAL CONTRAST
    contrast = librosa.feature.spectral_contrast(y = y,
                                                 sr = sr)
    
    #PAD/TRUNCATE
    mfcc = pad_or_turncate(mfcc)
    mfcc_delta = pad_or_turncate(mfcc_delta)
    mfcc_delta2 = pad_or_turncate(mfcc_delta2)
    mel = pad_or_turncate(mel)
    chroma = pad_or_turncate(chroma)
    zcr = pad_or_turncate(zcr)
    contrast = pad_or_turncate(contrast)

    #STACK FEATURES
    features = np.vstack([mfcc,
                          mfcc_delta,
                          mfcc_delta2,
                          mel,
                          chroma,
                          zcr,
                          contrast
    ])

    return features.astype(np.float32)