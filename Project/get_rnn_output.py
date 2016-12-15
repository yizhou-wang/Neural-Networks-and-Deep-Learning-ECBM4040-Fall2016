import numpy
import numpy as np
from scipy.io import savemat
import os
from os import walk
import librosa

def get_phase(nFFT = 1024, winsize = 1024, hop = 512):
    # filelist = []
    # i = 0
    # rootDir = os.getcwd()
    # print rootDir
    # for dirName, subdirList, fileList in os.walk(rootDir):  
    #     for dirName in fileList:
    #         if dirName.endswith('.wav'):

				# # print '!!!!!!!!!!!!!'
				# # print rootDir+dirName

				audio, fs = librosa.core.load('abjones_1_01.wav', sr=16000, mono=False)
				s1 = audio[0,:] # singing
				s2 = audio[1,:] # music
				mixture = s1 + s2

				input_stft = librosa.stft(mixture, n_fft=nFFT, win_length = winsize,hop_length=hop)
				s1_stft = librosa.stft(s1,n_fft=nFFT, win_length = winsize,hop_length=hop)
				s2_stft = librosa.stft(s2,n_fft=nFFT, win_length = winsize,hop_length=hop)

				input_stft_100 = input_stft[:,0:100]
				s1_stft_100 = s1_stft[:,0:100]
				s2_stft_100 = s2_stft[:,0:100]

				s1_phase = numpy.angle(s1_stft_100)
				s2_phase = numpy.angle(s2_stft_100)

				# i = i + 1
				# print "validation set #",i

				return s1_phase, s2_phase


def output_wav(MagMat_true, MagMat_pred):

	print 'MagMat Size =', MagMat_true.shape
	print 'MagMat Size =', MagMat_pred.shape

	s1_phase, s2_phase = get_phase()

	voice_true = MagMat_true.T[0:513]
	backgroud_true = MagMat_true.T[513:1026]
	voice_pred = MagMat_pred.T[0:513]
	backgroud_pred = MagMat_pred.T[513:1026]

	savemat('../4_Output_Wav/voice_mag_true.mat', mdict={'voice_mag_true': voice_true})
	savemat('../4_Output_Wav/background_mag_true.mat', mdict={'background_mag_true': backgroud_true})
	savemat('../4_Output_Wav/voice_mag_pred.mat', mdict={'voice_mag_pred': voice_pred})
	savemat('../4_Output_Wav/background_mag_pred.mat', mdict={'background_mag_pred': backgroud_pred})
	savemat('../4_Output_Wav/voice_phase.mat', mdict={'voice_phase': s1_phase})
	savemat('../4_Output_Wav/background_phase.mat', mdict={'background_phase': s2_phase})

	# wav_true_voice = istft(voice_true*cmath.exp(complex(0,s1_phase)), 1024, 1024, 512)
	# print wav_true_voice

	# wav_write('output/train0_true_voice.wav', 16000, wav_true_voice*32767)
	# wav_true_background = istft(backgroud_true*s2_phase, 1024, 1024, 512)
	# wav_write('output/train0_true_background.wav', 16000, wav_true_background)
	# wav_pred_voice = istft(voice_pred*s1_phase, 1024, 1024, 512)
	# wav_write('output/train0_pred_voice.wav', 16000, wav_pred_voice)
	# wav_pred_background = istft(backgroud_pred*s2_phase, 1024, 1024, 512)
	# wav_write('output/train0_pred_background.wav', 16000, wav_pred_background)
	# # *32767

	return

