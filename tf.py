import numpy as np
import numpy.fft as nf
import scipy.io.wavfile as wf
import matplotlib.pyplot as mp
sample_rate, sigs = wf.read('../data/freq.wav')
sigs = sigs / 2 ** 15
times = np.arange(len(sigs)) / sample_rate
freqs = nf.fftfreq(sigs.size, 1 / sample_rate)
ffts = nf.fft(sigs)
pows = np.abs(ffts)
mp.figure('Time Domain', facecolor='lightgray')
mp.title('Time Domain', fontsize=20)
mp.xlabel('Time', fontsize=14)
mp.ylabel('Signal', fontsize=14)
mp.tick_params(labelsize=10)
mp.grid(linestyle=':')
mp.plot(times, sigs, c='dodgerblue',
	label='Signal=f(Time)')
mp.legend()
mp.figure('Frequency Domain', facecolor='lightgray')
mp.title('Frequency Domain', fontsize=20)
mp.xlabel('Frequency', fontsize=14)
mp.ylabel('Power', fontsize=14)
mp.tick_params(labelsize=10)
mp.grid(linestyle=':')
mp.plot(freqs[freqs>=0], pows[freqs>=0],
	c='orangered', label='Power=F(Frequency)')
mp.legend()
mp.show()