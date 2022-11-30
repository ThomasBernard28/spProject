import scipy.signal as signal
import matplotlib.pyplot as plt
import numpy as np

def create_filter_cheby(wp, ws, gpass, gstop, fs):
    # create an anti-aliasing filter that eliminates all frequencies above 8000Hz

    # your code here #
    B, A = signal.cheby1(8, gpass, wp, 'low', analog=False, fs=fs)


    w, h = signal.freqz(B, A, fs=fs)
    plt.plot(0.5 * np.pi * w, 20 * np.log10(abs(h)), label="Chebychev I")  # pi/2 * w pour avoir la fréquence en Hz
    plt.legend(loc='upper right')
    plt.show()
    return B, A

#create_filter_cheby(8000, 10000, 1, 60, 44100)

def create_filter_cauer(wp, ws, gpass, gstop, fs):

    # your code here #
    B, A = signal.ellip(8, wp, ws, gstop, btype="lowpass", analog=False, output="ba", fs=fs)

    return B, A

b, a = create_filter_cauer(0.1, 60, 0, 8000, 44100)
w, h = signal.freqz(b, a, 2048, fs=44100)
h = 20 * np.log10(abs(h))
plt.plot(w, h)
plt.show()