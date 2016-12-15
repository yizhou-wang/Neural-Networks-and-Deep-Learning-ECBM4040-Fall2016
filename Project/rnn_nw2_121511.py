import theano
import timeit
import inspect
import sys
import mir_eval
from collections import OrderedDict
import numpy
import numpy as np
import theano.tensor as T
import scipy
from scipy import io
import librosa
from scipy.io import savemat
from load_data_norm import *
from scipy.io.wavfile import write
#sys.setrecursionlimit(3000)


# def load_data():
# 	data_x, data_y, phase = Input_data()
	
# 	valid_set_x = data_x[0]
# 	train_set_x = data_x[1]
# 	test_set_x = data_x[2]
	
	
# 	valid_set_y = data_y[0]
# 	train_set_y = data_y[1]
# 	test_set_y = data_y[2]
	
# 	return data_x, data_y
	
# def Mask(y1_pred, y2_pred):
# 	m_f = abs(y1_pred) / (abs(y1_pred) + abs(y2_pred))        
# 	return m_f


def istft_2_wav(train_y_pred, phase):

	# ISTFT for train_y_pred
	train_phase = phase[1][0]
	train_phase_exp = np.exp(1j * train_phase)

	print 'train_phase_exp.shape', train_phase_exp.shape

	train_y1_pred_stft = train_y_pred[0,:,0:513].T * train_phase_exp[0:513,:]
	train_y2_pred_stft = train_y_pred[0,:,513:1026].T * train_phase_exp[513:1026,:]

	print 'train_y1_pred_stft.shape =', train_y1_pred_stft.shape
	print 'train_y1_pred_stft.shape =', train_y1_pred_stft.shape

	train_y1_pred_istft = librosa.istft(
		train_y1_pred_stft, 
		win_length=1024,
		hop_length=512
	)
	train_y2_pred_istft = librosa.istft(
		train_y2_pred_stft, 
		win_length=1024,
		hop_length=512
	)
	print '\n!!!!! saving mat !!!!!\n'
	# savemat('../4_Output_Wav/train_y1_pred_istft.mat', 
	# 	mdict={'train_y1_pred_istft': train_y1_pred_istft})
	# savemat('../4_Output_Wav/train_y2_pred_istft.mat', 
	# 	mdict={'train_y2_pred_istft': train_y2_pred_istft})

	write('train_y1_pred_istft.wav', 16000, train_y1_pred_istft)
	write('train_y2_pred_istft.wav', 16000, train_y2_pred_istft)

	return


def mse(y_pred, y):
	
	y_pred_aug = 20 * np.log10(y_pred + 1e-3)
	y_aug = 20 * np.log10(y + 1e-3)
	return T.mean((y_pred_aug - y_aug) ** 2 )
	
def kl(self, y1, y2):
	pass

def contextwin(l, win, length):
	"""
	Return a list of list of indexes corresponding to context windows
	surrounding each word in the sentence

	:type win: int
	:param win: the size of the window given a list of indexes composing a sentence

	:type l: list or numpy.array
	:param l: array containing the word indexes

	"""
	assert (win % 2) == 1
	assert win >= 1
	l = list(l)

	lpadded = 513 * (win // 2) * [-1] + l + 513 * (win // 2) * [-1]
	# print '????????'
	# print lpadded

	# out = [lpadded[i*513:(i*513 + win*513)] for i in range(len(l)//513)]
	out = [lpadded[i*513:(i*513 + win*513)] for i in range(length)]
	# print len(out)
	# print np.array(out).shape

	assert len(out) == length
	return out
	
def RMSprop(params, grads, learning_rate, rou):
	updates = []
	epsilon = theano.shared(np.cast[theano.config.floatX](1e-6), name = 'epsilon')
	for p,g in zip(params, grads):
		r = theano.shared( p.get_value() * np.cast[theano.config.floatX](0.))
		r_new = rou * r + (np.cast[theano.config.floatX](1.) - rou) * g**2
		updates.append((r, r_new))

		p_update = -learning_rate * g / T.sqrt(r_new + epsilon)
		updates.append((p, p+p_update))
	return updates

class RNN(object):
	
	def __init__(self, n_hidden1, n_hidden2, dim_x, win_size, lr, rou):
		"""Initialize the parameters for the RNNSLU

		:type n_hidden1: int
		:param n_hidden: dimension of the hidden layer, number of hidden neuron

		:type n_hidden2: int
		:param n_hidden2: n_hidden of full-connected layer

		:type ne: int
		:param ne: number of word embeddings in the vocabulary

		:type dim_x: int
		:param dim_x: demention of input data

		:type win_size: int
		:param win_size: window context size

		"""
		# parameters of the model
		self.wx1 = theano.shared(name='wx',
								value=0.2 * numpy.random.uniform(-1.0, 1.0,
								(dim_x * win_size, n_hidden1))
								.astype(theano.config.floatX))
		self.wh1 = theano.shared(name='wh',
								value=0.2 * numpy.random.uniform(-1.0, 1.0,
								(n_hidden1, n_hidden1))
								.astype(theano.config.floatX))
		self.wx2 = theano.shared(name='wx2',
								value=0.2 * numpy.random.uniform(-1.0, 1.0,
								(n_hidden1, n_hidden1))
								.astype(theano.config.floatX))
		self.wh2 = theano.shared(name='wh2',
								value=0.2 * numpy.random.uniform(-1.0, 1.0,
								(n_hidden1, n_hidden1))
								.astype(theano.config.floatX))
		self.wx3 = theano.shared(name='wx3',
								value=0.2 * numpy.random.uniform(-1.0, 1.0,
								(n_hidden1, n_hidden1))
								.astype(theano.config.floatX))
		self.wh3 = theano.shared(name='wh3',
								value=0.2 * numpy.random.uniform(-1.0, 1.0,
								(n_hidden1, n_hidden1))
								.astype(theano.config.floatX))
		self.bh1 = theano.shared(name='bh',
								value=numpy.zeros(n_hidden1,
								dtype=theano.config.floatX))
		self.bh2 = theano.shared(name='bh2',
								value=numpy.zeros(n_hidden1,
								dtype=theano.config.floatX))
		self.bh3 = theano.shared(name='bh3',
								value=numpy.zeros(n_hidden1,
								dtype=theano.config.floatX))
		self.w = theano.shared(name='w',
							   value=0.2 * numpy.random.uniform(-1.0, 1.0,
							   (n_hidden1, n_hidden2 * dim_x))
							   .astype(theano.config.floatX))
		self.w2 = theano.shared(name='w2',
							   value=0.2 * numpy.random.uniform(-1.0, 1.0,
							   (n_hidden2 * dim_x, n_hidden2 * dim_x))
							   .astype(theano.config.floatX))
		self.b = theano.shared(name='b',
							   value=numpy.zeros(n_hidden2 * dim_x,
							   dtype=theano.config.floatX))
		self.b2 = theano.shared(name='b',
							   value=numpy.zeros(n_hidden2 * dim_x,
							   dtype=theano.config.floatX))
		self.h0 = theano.shared(name='h0',
								value=numpy.zeros(n_hidden1,
								dtype=theano.config.floatX))
		self.h1 = theano.shared(name='h1',
								value=numpy.zeros(n_hidden1,
								dtype=theano.config.floatX))
		self.h2 = theano.shared(name='h2',
								value=numpy.zeros(n_hidden1,
								dtype=theano.config.floatX))

		# bundle
		self.params = [self.wx1, self.wh1, self.wx2, self.wh2, self.wx3, self.wh3,
					   self.w2, self.b2, self.bh1, self.bh2, self.bh3, self.w, 
					   self.b, self.h0, self.h1, self.h2]

	   
		x = T.matrix()
		y = T.matrix()
		#y1 = T.ivector('y2')
		#y2 = T.ivector('y2')   #label
		lr = T.scalar('lr')  # learning rate

		def recurrence(x_t, h_tm1, h_tm2, h_tm3):
			h_t = T.tanh(T.dot(x_t, self.wx1) + T.dot(h_tm1, self.wh1) + self.bh1)
			h2_t = T.tanh(T.dot(h_t, self.wx2) + T.dot(h_tm2, self.wh2) + self.bh2)
			h3_t = T.tanh(T.dot(h2_t, self.wx3) + T.dot(h_tm3, self.wh3) + self.bh3)
			fc = T.tanh(T.dot(h3_t, self.w) + self.b)
			fc_2 = T.nnet.softmax(T.dot(fc, self.w2) + self.b2)
			#s_t = T.nnet.softmax(T.dot(fc, self.w) + self.b)
			#s1_t =T.nnet.sigmoid(T.dot(fc, self.ws1)+self.bs1)
			#s2_t =T.nnet.sigmoid(T.dot(fc, self.ws2)+self.bs2)
						#return [h_t, h2_t, h3_t, s_t]
			return [h_t, h2_t, h3_t, fc, fc_2]

		# [h, h2, h3, s], _ = theano.scan(fn=recurrence,
		#                         sequences=x,
		#                         outputs_info=[self.h0, self.h1, self.h2, None],
		#                         n_steps=x.shape[0])

		[h, h2, h3, fc, fc_2], _ = theano.scan(fn=recurrence,
								sequences=x,
								outputs_info=[self.h0, self.h1, self.h2, None, None],
								n_steps=x.shape[0])

		#h3 = T.tanh()
		# logicregression
		# p_y_given_x = s[:, 0, :]    # output
		# y_pred = T.argmax(p_y_given_x, axis=1) 
		# print "fc type", type(fc)
		# print "fc matrix", fc.get_value().shape
		
		#y_pred = fc[:, 0, :]    # output
		y_pred = fc_2[:,0,:]
		# negative_log_likelihood
		#nll = -T.mean(T.log(p_y_given_x)[T.arange(x.shape[0]), y])
		
		#cost
		cost = mse(y_pred, y)                     
	   
		grads = T.grad(cost, self.params)


		updates = OrderedDict((p, p - lr*g)
		                                for p, g in
		                                zip(self.params, grads))

		#updates = RMSprop(self.params, grads, lr, rou)
		
		self.test_model = theano.function(inputs=[x], outputs=y_pred, allow_input_downcast=True)
		self.train_model = theano.function(inputs=[x, y, lr],
											  outputs=cost,
											  updates=updates,
											  allow_input_downcast=True)

	def train(self, input_x, input_y, lr):

		#cinput = contextwin(x, win_size)
		inputs_x = list(map(lambda x: numpy.asarray(x).astype('float64'), input_x))
		inputs_y = list(map(lambda x: numpy.asarray(x).astype('float64'), input_y))
		# print np.array(inputs_x).shape
		# print np.array(inputs_y).shape

		self.train_model(inputs_x, inputs_y, lr)



def test (n_epoch, lr, n_hidden1, n_hidden2, win_size, rou, verbose=True, decay=True):
	'''
	:type n_epoch: int
	:param n_epoch: number of epochs

	:type lr: float
	:param lr: learning rate

	:type n_hidden: array
	:param n_hidden: [number of hiddens in each rnn layer, number of hiddens in fully-connected layer]

	:type win_size: int
	:param win_size: context window

	:type num: array
	:param num: number of train_sample, valid_sample, test_sample

	:type verbose: boolean
	:param verbose: to print out epoch summary or not to.

	'''
	print('... loading data')

	#train_set, valid_set, test_set = load_data()
	data_x, data_y, phase = Input_data()
	
 	valid_x = data_x[0].transpose(0,2,1)
 	train_x = data_x[1].transpose(0,2,1)
 	test_x = data_x[2].transpose(0,2,1)
		
 	valid_y = data_y[0].transpose(0,2,1)
 	train_y = data_y[1].transpose(0,2,1)
 	test_y = data_y[2].transpose(0,2,1)

	# train_x = train_set_x['Maginitude_Matrix'].transpose(0,2,1)	
	# valid_x = valid_set_x['Maginitude_Matrix'].transpose(0,2,1)	
	# test_x = test_set_x['A'].transpose(0,2,1)	
	#numpy.savetxt('train_x.csv', train_x, delimiter = ',')  

	print "dim of train_x is ", train_x.shape  #(171, 513, 400)
	print "dim of valid_x is ", valid_x.shape  #(4, 513, 400)
	print "dim of test_x is ", test_x.shape    #(825, 513, 400)

	

	# train_y = train_set['Maginitude_Matrix']
	# valid_y = valid_set['Maginitude_Matrix']
	# test_y = test_set['A']

	# train_y = np.hstack((train_set['person_voice'], train_set['background_voice']))


	print "... reshape train sets"
	train_x = train_x.reshape((-1, 513*100))[:1]
	valid_x = valid_x.reshape((-1, 513*100))[:1]
	test_x = test_x.reshape((-1, 513*100))[:1]

	#numpy.savetxt('train_x_reshape.csv', train_x, delimiter = ',')  


	print "dim of train_x is ", train_x.shape  
	print "dim of valid_x is ", valid_x.shape 
	print "dim of test_x is ", test_x.shape  

	
	# train_y = numpy.hstack((train_set['Maginitude_Matrix'], train_set['Maginitude_Matrix']))
	# train_y = train_y.reshape((-1, 513*400*2))[:3]

	# valid_y = numpy.hstack((valid_set['Maginitude_Matrix'], valid_set['Maginitude_Matrix']))
	# valid_y = valid_y.reshape((-1, 513*400*2))[:3]

	#train_y = train_set_y['Original_Matrix'].transpose(0,2,1)
	train_y = train_y.reshape((-1, 513*100*2))[:1]
	#numpy.savetxt('train_y_reshape.csv', train_y, delimiter = ',')  
	print train_y.shape

	#valid_y = valid_set_y['Original_Matrix'].transpose(0,2,1)
	valid_y = valid_y.reshape((-1, 513*100*2))[:1]


	#test_y = numpy.hstack((test_set_x['A'], test_set_x['A'])).transpose(0,2,1)
	test_y = test_y.reshape((-1, 513*100*2))[:1]

	print "dim of train_y is ", train_y.shape  
	print "dim of valid_y is ", valid_y.shape  
	print "dim of test_y is ", test_y.shape  


	print "*********---train_y----*********"
	print train_y

	# dim of train_x_org: (num[0], 513 * dim_x)
	# dim of train_x: (num[0] * dim_x, 513 * 3)
	# train_x = contextwin(train_x_org, window_size)
	# valid_x = contextwin(valid_x_org, window_size)
	# test_x = contextwin(test_x_org, window_size)
	# train_y = contextwin(train_y_org, window_size)
	# valid_y = contextwin(valid_y_org, window_size)
	# test_y = contextwin(test_y_org, window_size)

	print('... building the model')
	
	rnn = RNN(
		n_hidden1=n_hidden1, 
		n_hidden2=n_hidden2, 
		dim_x=513, 
		win_size=win_size,
		lr=lr,
		rou=rou
	)

	print('... training')
	best_validation_loss = numpy.inf
	clr = lr
	# record current learning rate 

	for epoch in range(n_epoch):

		tic = timeit.default_timer()

		for i, (x, y) in enumerate(zip(train_x, train_y)):
			# :type x: ndarray
			# :type y: ndarray
			#rnn.train(numpy.asarray(x), numpy.asarray(y), win_size, lr)
			# print type(x),type(y)
			# print x
			x = contextwin(x, 3, 100)
			y = y.reshape(100,-1)
			print 'training iteration', i

			rnn.train(x, y, lr)
			# if (i == 5):
			# 	print 'wyzwyzwyzwyz!!!!!!!'
			# 	break

			

		# evaluation // back into the real world
		# train_y_pred = np.asarray([rnn.test_model(x) for x in train_x])
		# valid_y_pred = np.asarray([rnn.test_model(x) for x in valid_x])
		# test_y_pred = np.asarray([rnn.test_model(x) for x in test_x])
		# test1 = (contextwin(x, win_size, 100) for x in train_x)
		# print "test1 shape***************"
		# print test1.shape

		train_y_pred = np.asarray([rnn.test_model(contextwin(x, win_size, 100)) for x in train_x])
		valid_y_pred = np.asarray([rnn.test_model(contextwin(x, win_size, 100)) for x in valid_x])
		test_y_pred = np.asarray([rnn.test_model(contextwin(x, win_size, 100)) for x in test_x])

		print "********--- train_y_pred ---**********"
		print train_y_pred.shape
		print "******************"
		#print train_y_pred[0]

		# istft_2_wav(train_y_pred, phase)

		
		train_y_pred_aug = 20 * np.log10(train_y_pred + 1e-2)
		valid_y_pred_aug = 20 * np.log10(valid_y_pred + 1e-2)
		test_y_pred_aug = 20 * np.log10(test_y_pred + 1e-2)

		#print "*********---train_y_pred----*********"
		#print train_y_pred_aug
		

		train_y_truth = np.asarray([y.reshape(100,-1) for y in train_y])
		valid_y_truth = np.asarray([y.reshape(100,-1) for y in valid_y])
		test_y_truth = np.asarray([y.reshape(100,-1) for y in test_y])

		# istft_2_wav(train_y_truth, phase)

		
		train_y_truth_aug = 20 * np.log10(train_y_truth + 1e-2)
		valid_y_truth_aug = 20 * np.log10(valid_y_truth + 1e-2)
		test_y_truth_aug = 20 * np.log10(test_y_truth + 1e-2)

		#print "*********---train_y_truth_aug----*********"
		print train_y_truth_aug.shape
		print train_y_pred_aug.shape

		######### ---reshape for computing SDR---########
		train_y_truth_aug_sdr = 20 * np.log10(train_y + 1e-2)
		valid_y_truth_aug_sdr = 20 * np.log10(valid_y + 1e-2)
		test_y_truth_aug_sdr = 20 * np.log10(test_y + 1e-2)

		print "******-- y_for_sdr******"
		print train_y_truth_aug_sdr.shape  #(1,102600)

		train_y_pred_aug_sdr = train_y_pred_aug.reshape(-1, 1026*100)
		valid_y_pred_aug_sdr = valid_y_pred_aug.reshape(-1, 1026*100)
		test_y_pred_aug_sdr = test_y_pred_aug.reshape(-1, 1026*100)

		train_y_pred_sdr = train_y_pred.reshape(-1, 1026*100)
		valid_y_pred_sdr = valid_y_pred.reshape(-1, 1026*100)
		test_y_pred_sdr = test_y_pred.reshape(-1, 1026*100)

		print "******--- y_for_sdr ---******"
		print train_y_pred_aug_sdr.shape  #(1,102600)
		

		res_train = np.mean((train_y_pred_aug - train_y_truth_aug) ** 2)
		res_valid = np.mean((valid_y_pred_aug - valid_y_truth_aug) ** 2)
		res_test = np.mean((test_y_pred_aug - test_y_truth_aug) ** 2)

		##########--- use 20log10 (dB) ---#########
		# train_eval = mir_eval.separation.bss_eval_sources(train_y_truth_aug_sdr, 
		# 										   		  train_y_pred_aug_sdr, 
		# 										   		 compute_permutation=False
		# 										   		)
		# valid_eval = mir_eval.separation.bss_eval_sources(valid_y_truth_aug_sdr, 
		# 										   		  valid_y_pred_aug_sdr,
		# 										   		 compute_permutation=False
		# 										   		)
		# test_eval = mir_eval.separation.bss_eval_sources(test_y_truth_aug_sdr, 
		# 										   		 test_y_pred_aug_sdr,
		# 										   		 compute_permutation=False
		# 										   		)
		
		##########--- do not use 20log10 (dB) ---#########
		train_eval = mir_eval.separation.bss_eval_sources(train_y, 
												   		  train_y_pred_sdr, 
												   		 compute_permutation=False
												   		)
		valid_eval = mir_eval.separation.bss_eval_sources(valid_y, 
												   		  valid_y_pred_sdr,
												   		 compute_permutation=False
												   		)
		test_eval = mir_eval.separation.bss_eval_sources(test_y, 
												   		 test_y_pred_sdr,
												   		 compute_permutation=False
												   		)
		#print "**************--SDR--*************"
		train_sdr = train_eval[0]
		valid_sdr = valid_eval[0]
		test_sdr = test_eval[0]
		print train_sdr.shape

		train_sdr_mean = np.mean(train_sdr)
		valid_sdr_meam = np.mean(valid_sdr)
		test_sdr_mean = np.mean(test_sdr)

		train_sdr_max = np.amax(train_sdr)
		valid_sdr_max = np.amax(valid_sdr)
		test_sdr_max = np.amax(test_sdr)

		train_sdr_min = np.amin(train_sdr)
		valid_sdr_min = np.amin(valid_sdr)
		test_sdr_min = np.amin(test_sdr)

	

		if (verbose):
				print ('epoch %i, valid SDR %f' % (epoch, valid_sdr_meam))

		
		if res_valid < best_validation_loss:
			best_score = res_valid

			if (verbose):
				print ('NEW BEST: epoch %i, training SDR %f , valid SDR %f , best test SDR %f ' %
					   (epoch, train_sdr_mean, valid_sdr_meam, test_sdr_mean))
				print ('max training SDR %f , max valid SDR %f , max test SDR %f ' %
					   (train_sdr_max, valid_sdr_max, test_sdr_max))
				print ('min training SDR %f , min valid SDR %f , min test SDR %f ' %
					   (train_sdr_min, valid_sdr_min, test_sdr_min))
			
			
			# record current res_valid and res_test
			vf1, tf1 = valid_sdr_meam, test_sdr_mean 
			be = epoch

		if (decay) and abs(be - epoch) >= 10:
			clr *= 0.5
			

		if clr < 1e-5:
			break

	print('BEST RESULT: epoch %i, valid SDR %f, best test SDR %f' %
		  (be, vf1, tf1))
	
	istft_2_wav(train_y_pred, phase)


if __name__ == '__main__':
	test (n_epoch = 100, lr = 0.05, n_hidden1=1000, n_hidden2=2, win_size = 3, rou = 0.8, verbose=True, decay=True)
	





