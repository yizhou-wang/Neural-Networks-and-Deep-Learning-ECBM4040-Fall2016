import librosa
import scipy
from scipy.io.wavfile import read,write
from scipy.io import wavfile
import numpy
import numpy as np
import os
from os import walk
import matplotlib.pyplot as plt

def load_data_1(dir, wavnum, nFFT = 1024, winsize = 1024, hop = 512):
    filelist = []
    Magnitude_Matrix = np.zeros([wavnum,513,100])
    Phase_Matrix = np.zeros([wavnum,1026,100])
    Original_Matrix = np.zeros([wavnum,1026,100])
    # saved_column_dev = np.zeros([4,1])
    i = 0
    #print dir
    rootDir = 'Wavfile/' + dir + '/'
    #print rootDir
    for dirName, subdirList, fileList in os.walk(rootDir):  
        for dirName in fileList:
            if dirName.endswith('.wav'):
                audio, fs = librosa.core.load(rootDir+dirName, sr=16000, mono=False)
                # print "audio", audio
                # print "fs", fs
                # print audio.shape
                s1 = audio[0,:] # singing
                # print "s1.shape", s1.shape
                s2 = audio[1,:] # music
                s1 = (s1 - numpy.mean(s1))/numpy.std(s1)
                s2 = (s2 - numpy.mean(s2))/numpy.std(s2)
                mixture = s1 + s2
                # print "mixture.shape",mixture.shape
                input_stft = librosa.stft(mixture, n_fft=nFFT, win_length = winsize,hop_length=hop)
                per_batch, column_number = input_stft.shape
                #print "column_number",column_number

                # print "input_stft", input_stft.shape
                s1_stft = librosa.stft(s1,n_fft=nFFT, win_length = winsize,hop_length=hop)
                s2_stft = librosa.stft(s2,n_fft=nFFT, win_length = winsize,hop_length=hop)
                # print "input_stft_100", input_stft_100[0,0]
                # LogMagnitude = librosa.logamplitude(np.abs(input_stft_100)**2,ref_power=np.max)
                # print "LogMagnitude", LogMagnitude.shape
                # print "LogMagnitude", LogMagnitude[0,0]
                #S = np.abs(librosa.stft(s1))
                # plt.figure()
                # librosa.display.specshow(LogMagnitude, sr=fs, y_axis='log', x_axis='time')
                # plt.colorbar()
                # plt.title('Power spectrogram')
                # plt.show()
                # print "s1_stft_100.shape", s1_stft_100.shape
                # print "s1_stft",s1_stft.shape
                # print "s2_stft",s2_stft.shape
                if column_number < 100:
	                zeropad_size = 100-column_number
                	zero_pad = numpy.zeros((per_batch,zeropad_size))
                	input_stft_100 = numpy.column_stack((input_stft,zero_pad))
                	s1_stft_100 = numpy.column_stack((s1_stft,zero_pad))
                	s2_stft_100 = numpy.column_stack((s2_stft,zero_pad))
                else:
	                input_stft_100 = input_stft[:,0:100]
                	s1_stft_100 = s1_stft[:,0:100]
                	s2_stft_100 = s2_stft[:,0:100]
                # s1_stft_zeropad = numpy.column_stack((s1_stft,zero_pad))
                # s2_stft_zeropad = numpy.column_stack((s2_stft,zero_pad))
                # input_phase = numpy.angle(input_stft_100)
                # input_phase_zeropad = numpy.column_stack((input_phase,zero_pad))
                # saved_column_dev[i,:] = column_number
                Magnitude_Matrix[i,:,:] = abs(input_stft_100)
                Phase_Matrix[i,:,:] = numpy.row_stack((numpy.angle(s1_stft_100),numpy.angle(s2_stft_100)))
                Original_Matrix[i,:,:] = numpy.row_stack((abs(s1_stft_100),abs(s2_stft_100)))
                # s1_stft = np.zeros([513,100])
                # s2_stft_zeropad = np.zeros([513,100])
                # input_stft_zeropad = np.zeros([513,100])
                # input_phase_zeropad = np.zeros([513,100])
                i = i + 1
                #print "Set #",i

    return Magnitude_Matrix, Phase_Matrix, Original_Matrix

def Input_data(nFFT = 1024, winsize = 1024, hop = 512):

	Magnitude_Matrix_dev, Phase_Matrix_dev, Original_Matrix_dev = load_data_1('dev', 4)
	Magnitude_Matrix_train, Phase_Matrix_train, Original_Matrix_train = load_data_1('train', 171)
	Magnitude_Matrix_test, Phase_Matrix_test, Original_Matrix_test = load_data_1('test', 825)
	train_x = [Magnitude_Matrix_dev, Magnitude_Matrix_train, Magnitude_Matrix_test]
	train_y = [Original_Matrix_dev, Original_Matrix_train, Original_Matrix_test]
	phase = [Phase_Matrix_dev, Phase_Matrix_train, Phase_Matrix_test]
	# var_return = (Magnitude_Matrix_dev, Original_Matrix_dev, Phase_Matrix_dev,
	# 	Magnitude_Matrix_train, Original_Matrix_train, Phase_Matrix_train,
	# 	Magnitude_Matrix_test, Original_Matrix_test, Phase_Matrix_test
	# )
	return train_x, train_y, phase

if __name__ == '__main__':
    Input_data()

