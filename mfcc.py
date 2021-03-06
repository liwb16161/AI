import numpy as np
import scipy.io.wavfile as wf
import python_speech_features as sf
import matplotlib.pyplot as mp
sample_rate, sigs = wf.read(
	'D:/ML/data/speeches/training/banana/banana01.wav')
mfcc = sf.mfcc(sigs, sample_rate)
mp.matshow(mfcc.T, cmap='gist_rainbow',
	fignum='MFCC')
mp.title('MFCC', fontsize=20)
mp.xlabel('Sample', fontsize=14)
mp.ylabel('Feature', fontsize=14)
mp.tick_params(which='both', top=False,
	labeltop=False, labelbottom=True,
	labelsize=10)
mp.show()