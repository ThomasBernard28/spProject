# %% [markdown]
# # Signal Processing Project: real-time sound localisation

# %% [markdown]
# ## 1 Offline system
# ### 1.1 Data generation and dataset

# %%
import numpy as np
import matplotlib.pyplot as plt


def create_sine_wave(f, A, fs, N):
    # your code here #
    sample = np.arange(N)
    t = sample / fs
    out = A * np.sin(2 * np.pi * f * t)

    return out


# call and test your function here #
fs = 44100
N = 8000
freq = 20
amplitude = 8 / 2
# plt.plot(np.arange(N)/fs, create_sine_wave(freq, amplitude, fs, N))
# plt.show()


# %%
from glob import glob
import scipy.io.wavfile as wf


def read_wavefile(path):
    fs, data = wf.read(path)

    return fs, data


# call and test your function here #
LocateClaps = "LocateClaps"
files = glob(f"{LocateClaps}/*.wav")

result = read_wavefile(files[0])

# plt.plot(result[1])
# plt.show()


# %% [markdown]
# ### 1.2 Buffering

# %%
from collections import deque


def create_ringbuffer(maxlen):
    out = deque(maxlen=maxlen)

    return out


# call and test your function here #
stride = 750
maxlen = 750

ringbuffer = create_ringbuffer(maxlen)

# reading your signal as a stream:
for i, sample in enumerate(create_sine_wave(20, 8 / 2, 44100, 8000)):
    ringbuffer.append(sample)
    if i == maxlen:
        # plt.plot(ringbuffer)
        # plt.show()
        pass

    if i == 1749:
        # plt.plot(np.arange(stride) + 1000 ,ringbuffer)
        # plt.show()
        pass

    # your code here #


# %% [markdown]
# ### 1.3 Pre-processing
# #### 1.3.1 Normalisation

# %%
def normalise(s):
    # your code here #
    return s / np.max(np.abs(s))


# call and test your function here #
def test():
    # test function
    s = create_sine_wave(20, 8 / 2, 44100, 8000)
    plt.plot(np.arange(8000) / 44100, s, color='red', label="wave form")
    plt.legend(loc='upper right')

    s_norm = normalise(s)
    plt.plot(np.arange(8000) / 44100, s_norm, color='blue', label="normalised wave form")
    plt.legend(loc='upper right')

    plt.show()


# test()

# %% [markdown]
# #### 1.3.2 Downsampling

# %%
import scipy.signal as signal

## 1 - spectral analysis via spectrogram
LocateClaps = "LocateClaps/M1_0.wav"
wavefile = read_wavefile(LocateClaps)

plt.specgram(wavefile[1], Fs=wavefile[0])
plt.yticks(np.arange(0, 23000, 1000))
plt.title("Spectrogram")
plt.show()


# On voit que les fréquences utiles vont jusque 8000Hz.
# On va donc passer de 44.1kHz à 2 * 8kHz = 16kHz

## 2 - Anti-aliasing filter synthesis


def create_filter_cheby(wp, ws, gpass, gstop, fs):
    # your code here #
    N, wn = signal.cheb2ord(wp, ws, gpass, gstop, fs=fs)
    B, A = signal.cheby2(N, gstop, wn, "low", fs=fs)

    return B, A


def create_filter_cauer(wp, ws, gpass, gstop, fs):
    # your code here #
    N, wc = signal.ellipord(wp, ws, gpass, gstop, fs=fs)
    B, A = signal.ellip(N, gpass, gstop, wc, "low", fs=fs)

    return B, A


## 3 - Decimation
def downsampling(sig, B, A, M):
    filtered = signal.lfilter(B, A, sig)
    out = filtered[::M]

    return out


wp = 8000
ws = 8200
gpass = 0.1
gstop = 60

# call and test your function here
b, a = create_filter_cauer(wp, ws, gpass, gstop, 44100)

w, h = signal.freqz(b, a, 2048, fs=44100)
h = 20 * np.log10(abs(h))
# plt.plot(w, h)
# plt.title("Cauer filter")
# plt.show()


sin = create_sine_wave(8500, 1000, 44100, 8000) + create_sine_wave(7500, 20, 44100, 8000)
# plt.plot(np.arange(8000)/44100,sin, color="blue",label="original sin wave 1" )
# plt.legend("upper right")
# plt.title("Sine wave before the filter")
# plt.show()

filtered1 = signal.lfilter(b, a, sin)
# plt.plot(np.arange(8000)/44100,filtered1, color="red", label="filtered wave 1")
# plt.legend("upper right")
# plt.title("Sine wave after the filter")
# plt.show()

decimated = downsampling(sin, b, a, 3)
# plt.plot(decimated)
# plt.title("Decimation")
# plt.show()

# %% [markdown]
# ### 1.4 Cross-correlation

# %%
## 1.4
import scipy.signal as sc
import numpy as np


def fftxcorr(in1, in2):
    # Utilise fft de numpy
    # il y a un paramètre N a donner (important)

    # your code here #
    out = np.fft.ifft(np.fft.fft(in1, 2 * len(in1)) * np.fft.fft(in2[::-1], 2 * len(in2)))
    return out.real


# call and test your function here #
fs = 44100
N = 8000
freq = 20
amplitude = 8 / 2
sin1 = create_sine_wave(freq, amplitude, fs, N)

sin2 = create_sine_wave(50, amplitude, fs, N)

# plt.plot(fftxcorr(sin1, sin2), color="blue")
xcorr_fftconv = sc.fftconvolve(sin1, sin2[::-1], 'full')  # [::-1] flips the signal but you can also use np.flip()


# plt.plot(xcorr_fftconv, color="red")

# %% [markdown]
# ### 1.5 Localisation
# #### 1.5.1 TDOA

# %%
def TDOA(xcorr):
    out = np.argmax(xcorr)
    return (out - len(xcorr) / 2)


# %% [markdown]
# #### 1.5.2 Equation system

# %%
from scipy.optimize import root

# mic coordinates in meters
MICS = [{'x': 0, 'y': 0.0487}, {'x': 0.0425, 'y': -0.025}, {'x': -0.0425, 'y': -0.025}]


def equations(p, deltas):
    v = 343
    x, y = p
    eq1 = v * deltas[0] - np.sqrt((MICS[0]['x'] - x) ** 2 + (MICS[0]['y'] - y) ** 2) + np.sqrt(
        (MICS[1]['x'] - x) ** 2 + (MICS[1]['y'] - y) ** 2)
    eq2 = v * deltas[1] - np.sqrt((MICS[0]['x'] - x) ** 2 + (MICS[0]['y'] - y) ** 2) + np.sqrt(
        (MICS[2]['x'] - x) ** 2 + (MICS[2]['y'] - y) ** 2)
    return eq1, eq2


def localize_sound(deltas):
    sol = root(equations, [0, 0], deltas, tol=10)
    return sol.x


def source_angle(coordinates):
    # your code here
    x = coordinates[0]  # x of the source
    y = coordinates[1]  # y of the source

    out = np.arctan(y / x) * 180 / np.pi

    if out < 0:
        if y > 0:
            out = 180 + out
        else:
            out = 360 + out
    elif out > 0:
        if y < 0:
            out = 180 + out

    return out


# call and test your function here #
coord = (-5, 5)
print(f"theta = {source_angle(coord)}")


# %% [markdown]
# ### 1.6 System accuracy and speed

# %%
## 1.6.1
def accuracy(pred_angle, gt_angle, threshold):
    # your code here #

    return abs(pred_angle - gt_angle) <= threshold


## 1.6.2
accuracyList = []
possible_angle = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
b, a = create_filter_cauer(1, 8000, 0.1, 60, 44100)
for angle in possible_angle:
    M1 = 0
    M2 = 0
    M3 = 0
    for f in files:
        if f'_{angle}.' in f:
            mic = f.split('/')[-1].split('_')[0]  # if '/' does not work, use "\\" (windows notation)
            if mic == 'M1':
                M1 = read_wavefile(f)[1]
            elif mic == 'M2':
                M2 = read_wavefile(f)[1]
            elif mic == 'M3':
                M3 = read_wavefile(f)[1]

    # call and test your function here #

    M1 = normalise(M1)
    M2 = normalise(M2)
    M3 = normalise(M3)

    # Rajouter le downsampling ici

    M1 = downsampling(M1, b, a, 2)
    M2 = downsampling(M2, b, a, 2)
    M3 = downsampling(M3, b, a, 2)

    xcorr1_2 = fftxcorr(M1, M2)
    xcorr1_3 = fftxcorr(M1, M3)

    tdoa2_1 = TDOA(xcorr1_2) / fs
    tdoa3_1 = TDOA(xcorr1_3) / fs

    sound = localize_sound([tdoa2_1, tdoa3_1])
    calculated_angle = round(source_angle(sound), 3)

    accuracyList.append(abs(calculated_angle - angle))

    # On va choisir un threshold de 14° car on voit que chaque test d'angle est séparé de 30°, ça nous permet de voir
    # qu'on ne confond pas un angle d'un autre.
    print(f"Angle = {angle} : {calculated_angle}. accuracy : {accuracy(angle, calculated_angle, 14)}.")

print(f"Moyenne de accuracy = {np.mean(accuracyList)}")

## 1.6.3
from time import time_ns, sleep


def func_example(a, b):
    return a * b


def time_delay(func, args):
    start_time = time_ns()
    out = func(*args)
    end_time = time_ns()
    print(f"{func.__name__} in {end_time - start_time} ns")
    return out


product = time_delay(func_example, [2, 10])

# call and test your previous functions here #

print("\n ====== Time test ====== \n")

fs, M1_30 = read_wavefile("LocateClaps/M1_60.wav")
M2_30 = read_wavefile("LocateClaps/M2_60.wav")[1]
M3_30 = read_wavefile("LocateClaps/M3_60.wav")[1]


def time_test(M1, M2, M3):
    time_delay(normalise, [M1])
    time_delay(downsampling, [M1, b, a, 3])

    time_delay(fftxcorr, [M1, M2])
    time_delay(sc.fftconvolve, [M1, M2[::-1], 'full'])

    time_delay(TDOA, [fftxcorr(M1, M2)])
    time_delay(localize_sound, [[TDOA(fftxcorr(M1, M2)) / fs, TDOA(fftxcorr(M1, M3)) / fs]])
    time_delay(source_angle, [localize_sound([TDOA(fftxcorr(M1, M2)) / fs, TDOA(fftxcorr(M1, M3)) / fs])])


print("==== Test without downsampling ====")
time_test(M1_30, M2_30, M3_30)

M1 = downsampling(M1, b, a, 2)
M2 = downsampling(M2, b, a, 2)
M3 = downsampling(M3, b, a, 2)

print("\n==== Test with downsampling ====")
time_test(M1, M2, M3)

# %% [markdown]
# ## 2 Real-time localisation

# %% [markdown]
# ### 2.1 Research one Raspberry Pi application

# %% [markdown]
# ### 2.2 Data acquisition and processing

# %%
#### Callback 
import pyaudio

RESPEAKER_CHANNELS = 8
BUFFERS = []


def callback(in_data, frame_count, time_info, flag):
    global BUFFERS
    data = np.frombuffer(in_data, dtype=np.int16)
    BUFFERS[0].extend(data[0::RESPEAKER_CHANNELS])
    BUFFERS[1].extend(data[2::RESPEAKER_CHANNELS])
    BUFFERS[2].extend(data[4::RESPEAKER_CHANNELS])
    return (None, pyaudio.paContinue)


#### Stream management

RATE = 44100
RESPEAKER_WIDTH = 2
CHUNK_SIZE = 2048


def init_stream():
    print("========= Stream opened =========")
    p = pyaudio.PyAudio()
    for i in range(p.get_device_count()):
        device_info = p.get_device_info_by_index(i)

        if device_info['maxInputChannels'] == 8:
            INDEX = i
            break

        if i == p.get_device_count() - 1:
            # Sound card not found
            raise OSError('Invalid number of channels')

    stream = p.open(rate=RATE, channels=RESPEAKER_CHANNELS, format=p.get_format_from_width(RESPEAKER_WIDTH), input=True,
                    input_device_index=INDEX,
                    frames_per_buffer=CHUNK_SIZE, stream_callback=callback)

    return stream


def close_stream(stream):
    print("========= Stream closed =========")
    stream.stop_stream()
    stream.close()


#### Detection and visual feedback
def detection(stream):
    global BUFFERS, pixel_ring

    if stream.is_active():
        print("========= Recording =========")

    while stream.is_active():
        try:
            if len(BUFFERS[0]) > CHUNK_SIZE:
                st = time_ns()
                deltas = [TDOA(fftxcorr(BUFFERS[0], BUFFERS[1])), TDOA(fftxcorr(BUFFERS[0], BUFFERS[2]))]

                x, y = localize_sound(deltas)
                hyp = np.sqrt(x ** 2 + y ** 2)

                ang_cos = round(np.arccos(x / hyp) * 180 / np.pi, 2)
                ang_sin = round(np.arcsin(y / hyp) * 180 / np.pi, 2)

                if ang_cos == ang_sin:
                    ang = ang_cos
                else:
                    ang = np.max([ang_cos, ang_sin])
                    if ang_cos < 0 or ang_sin < 0:
                        ang *= -1
                ang *= -1

                print((time_ns() - st) / 1e9, ang)

                print(np.max(BUFFERS, axis=-1))

                if (np.max(BUFFERS, axis=-1) > 3000).any():
                    pixel_ring.wakeup(ang)
                else:
                    pixel_ring.off()

                sleep(0.5)

        except KeyboardInterrupt:
            print("========= Recording stopped =========")
            break


#### Launch detection
from pixel_ring.apa102_pixel_ring import PixelRing
from gpiozero import LED

USED_CHANNELS = 3

power = LED(5)
power.on()

pixel_ring = PixelRing(pattern='soundloc')

pixel_ring.set_brightness(10)

for i in range(USED_CHANNELS):
    BUFFERS.append(create_ringbuffer(3 * CHUNK_SIZE))

stream = init_stream()

while True:
    try:
        detection(stream)
        sleep(0.5)
    except KeyboardInterrupt:
        break

close_stream(stream)

power.off()
