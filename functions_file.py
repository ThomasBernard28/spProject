import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import scipy.io.wavfile as wf
from collections import deque
import scipy.signal as signal
from scipy.optimize import root
from time import time_ns, sleep

import pyaudio
from pixel_ring.apa102_pixel_ring import PixelRing
from gpiozero import LED


def create_sine_wave(f, A, fs, N):
    # your code here #
    sample = np.arange(N)
    t = sample / fs
    out = A * np.sin(2 * np.pi * f * t)

    return out


def read_wavefile(path):
    fs, data = wf.read(path)

    return fs, data


LocateClaps = "LocateClaps"
files = glob(f"{LocateClaps}/*.wav")


def create_ringbuffer(maxlen):
    out = deque(maxlen=maxlen)

    return out


def normalise(s):
    # your code here #
    return s / np.max(np.abs(s))


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


def downsampling(sig, B, A, M):
    filtered = signal.lfilter(B, A, sig)
    out = filtered[::M]

    return out


def fftxcorr(in1, in2):
    # Utilise fft de numpy
    # il y a un paramètre N a donner (important)

    # your code here #
    out = np.fft.ifft(np.fft.fft(in1, 2 * len(in1)) * np.fft.fft(in2[::-1], 2 * len(in2)))
    return out.real


def TDOA(xcorr):
    out = np.argmax(xcorr)
    return out - len(xcorr) / 2


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


def accuracy(pred_angle, gt_angle, threshold):
    # your code here #

    return abs(pred_angle - gt_angle) <= threshold


def func_example(a, b):
    return a * b


def time_delay(func, args):
    start_time = time_ns()
    out = func(*args)
    end_time = time_ns()
    print(f"{func.__name__} in {end_time - start_time} ns")
    return out


# ### 2.2 Data acquisition and processing

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
