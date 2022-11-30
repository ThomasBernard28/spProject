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

create_filter_cheby(8000, 10000, 1, 60, 44100)