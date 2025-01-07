import librosa
from librosa import display
import numpy as np
import IPython.display as ipd
import matplotlib as plt
import soundfile as sf
import pedalboard as pdb
import noisereduce as nr
from pedalboard import Pedalboard, Compressor, Gain
board = Pedalboard()
cps = Compressor(threshold_db=-50, ratio=25)
gn = Gain(gain_db=30)
board.append(cps)
board.append(gn)
S_full, phase = librosa.magphase(librosa.stft(y))
idx = slice(*librosa.time_to_frames([100*130], sr=sr)) 
fig, ax = plt.pyplot.subplots()               
img = display.specshow(librosa.amplitude_to_db(S_full[:, idx], ref=np.max), y_axis='log', x_axis='time', sr=sr, ax=ax)
fig.colorbar(img, ax=ax)
S_filter = librosa.decompose.nn_filter(S_full, aggregate=np.median, metric='cosine', width=int(librosa.time_to_frames(2, sr=sr)))
S_filter = np.minimum(S_full, S_filter)
margin_i, margin_v = 3, 11  
power = 3
mask_i = librosa.util.softmask(S_filter, margin_i * (S_full - S_filter), power=power)
mask_v = librosa.util.softmask(S_full - S_filter, margin_v * S_filter, power=power)
S_background = mask_i * S_full
# Plot its foreground and background
fig, ax = plt.pyplot.subplots(nrows=3, sharex=True, sharey=True)
img = display.specshow(librosa.amplitude_to_db(S_full[:, idx], ref=np.max), y_axis='log', x_axis='time', sr=sr, ax=ax[0])
ax[0].set(title='Full Spectrum')
ax[0].label_outer()
display.specshow(librosa.amplitude_to_db(S_background[:, idx], ref=np.max), y_axis='log', x_axis='time', sr=sr, ax=ax[1])
ax[1].set(title='Background Spectrum')
ax[1].label_outer()
display.specshow(librosa.amplitude_to_db(S_foreground[:, idx], ref=np.max), y_axis='log', x_axis='time', sr=sr, ax=ax[2])
ax[2].set(title='Foreground Spectrum')
ax[2].label_outer()

fig.colorbar(img, ax=ax)
# Recover the foreground audio (vocals)
y_foreground = librosa.istft(S_foreground * phase)
sf.write('Vocals.wav', y_foreground, sr, subtype="PCM_24")
ipd.Audio(data=y_foreground, rate=sr)
data, samplerate = sf.read('Vocals.wav')
# reduce noise
y_reduced_noise = nr.reduce_noise(y=data, sr=samplerate)
sf.write('Vocals_reduced.wav', y_reduced_noise, samplerate, subtype="PCM_24")
data, samplerate = librosa.load('Vocals_reduced.wav')
ipd.Audio('Vocals_reduced.wav’)
# Recover the background audio (instrumental)
x_background = librosa.istft(S_background * phase)
sf.write('Instruments.wav', x_background, sr, subtype="PCM_24")
y, sr = librosa.load('Instruments.wav')
ipd.Audio('Instruments.wav')