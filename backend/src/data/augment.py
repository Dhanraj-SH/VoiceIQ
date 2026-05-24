import numpy as np
import librosa

def time_stretch(y, rate = 0.9):
    return librosa.effects.time_stretch(y , rate = rate)

def pitch_shift(y, sr, steps = 2):
    return librosa.effects.pitch_shift(y,
                                       sr = sr,
                                       n_steps = steps)

def add_noise(y, factor = 0.005):
    noise = np.random.randn(len(y))
    return y + factor * noise