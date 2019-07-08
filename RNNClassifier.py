from random import seed
seed(1)

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np

from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.models import load_model
from keras.layers import Dropout, LSTM, Dense
from keras import optimizers



class PreProcessing():
	"""Process the ICI data in the appropriate spreadsheet

	Parameters
	------------
	file_name : str
		Name of the CSV file
	ici_start : int
		Index of the column of the first ICI value
	ici_end : int
		Index of the column of the final ICI value
	coda_type_col : int
		Index of the column containing coda type labels
	clan_col : int 
		Index of the column containing clan labels
	whale_id_col : int
		Index of the column containing whale ID labels
	pretrain_test_size : float
		Fraction of dataset to be used for testing
	coda_removal : array-like
		Array of strings containing noise codas
	removal_treshold : int
		Number of codas per whale that must be present to include the whale ID

	Methods
	------------
	read_file
		Reads the CSV file
	pretrain_processing
		Processes the spreadsheet for pretraining the base model
	gen_ici_matrix
		Generates the matrix containing ICI values
	coda_type_processing
		Processes the spreadsheet to prepare for coda type classification
	vocal_clan_processing
		Processes the spreadsheet to prepare for vocal clan classification
	whale_id_processing
		Processes the spreadsheet to prepare for whale ID classification
	"""

	def __init__(self, file_name, ici_start=4, ici_end=13,
		coda_type_col=13, clan_col=14, whale_id_col=17, pretrain_test_size=0.1):
		self.file_name = file_name
		self.ici_start = ici_start
		self.ici_end = ici_end
		self.coda_type_col = coda_type_col
		self.clan_col = clan_col
		self.whale_id_col = whale_id_col
		self.pretrain_test_size = pretrain_test_size

	def read_file(self):
		"""Read the CSV file"""

		df_codas = pd.read_csv(self.file_name, header=0)
		return df_codas

	def pretrain_processing(self):
		"""Return training and testing datasets 
		for the initial sequence prediction task
		"""

		df_codas_ici = self.read_file().iloc[:,
			self.ici_start:self.ici_end]

		ici_matrix_data = df_codas_ici.values
		ici_matrix = np.zeros((ici_matrix_data.shape[0],
			ici_matrix_data.shape[1]))
		for i in range(ici_matrix_data.shape[0]):
			first_zero = 0
			for j in range(ici_matrix_data.shape[1]):
				if ici_matrix_data[i,j]==0:
					first_zero = j
					break
			ici_matrix[i,(ici_matrix_data.shape[1]-j):] = ici_matrix_data[i,0:j]

		x_ici_matrix = ici_matrix[:,0:-1]
		y_ici_matrix = ici_matrix[:,-1]

		x_train_net1, x_test_net1, y_train_net1, y_test_net1 = train_test_split(
			x_ici_matrix, y_ici_matrix, test_size=self.pretrain_test_size, 
			random_state=123)

		x_train_net1 = x_train_net1.reshape(x_train_net1.shape[0], 
			x_train_net1.shape[1], 1)
		x_test_net1 = x_test_net1.reshape(x_test_net1.shape[0], 
			x_test_net1.shape[1], 1)

		return x_train_net1, x_test_net1, y_train_net1, y_test_net1

	def gen_ici_matrix(self):
		"""Generate the matrix containing
		ICI values, ordered appropriately"""

		df_codas_ici = self.read_file().iloc[:,
			self.ici_start:self.ici_end]

		ici_matrix_data = df_codas_ici.values
		ici_matrix = np.zeros((ici_matrix_data.shape[0],
			ici_matrix_data.shape[1]))
		for i in range(ici_matrix_data.shape[0]):
			first_zero = 0
			for j in range(ici_matrix_data.shape[1]):
				if ici_matrix_data[i,j]==0:
					first_zero = j
					break
			ici_matrix[i,(ici_matrix_data.shape[1]-j):] = ici_matrix_data[i,0:j]
		return ici_matrix

	def coda_type_processing(self, coda_removal = ['1-NOISE', '2-NOISE',
		'3-NOISE','4-NOISE','5-NOISE','6-NOISE','7-NOISE','8-NOISE',
		'9-NOISE','10-NOISE','10i','10R']):
		"""Process the coda type data, making sure to remove
		the codas characterized as #-NOISE"""

		ici_matrix = self.gen_ici_matrix()
		coda_type_labels = self.read_file().iloc[:,self.coda_type_col].values
		unique_coda_type_labels = np.unique(coda_type_labels)

		x_coda_type_data = ici_matrix[:,1:]

		for noise in coda_removal:
			x_coda_type_data = x_coda_type_data[coda_type_labels!=noise]
			coda_type_labels = coda_type_labels[coda_type_labels!=noise]

		unique_coda_type_labels = np.unique(coda_type_labels)

		x_ct_data = np.asarray(x_coda_type_data)
		x_ct_data = x_ct_data.reshape(x_ct_data.shape[0], x_ct_data.shape[1], 1)
		y_ct_data = coda_type_labels
		unique_ct = unique_coda_type_labels

		class_mapping = {label:idx for idx,label in enumerate(unique_ct)}

		label_encoder = LabelEncoder()
		label_encoder.fit(y_ct_data)
		y_encoded_labels = label_encoder.transform(y_ct_data)
		y_one_hot = np_utils.to_categorical(y_encoded_labels)

		return class_mapping, unique_ct, x_ct_data, y_one_hot

	def vocal_clan_processing(self):
		ici_matrix = self.gen_ici_matrix()
		clan_labels = self.read_file().iloc[:,self.clan_col].values
		unique_clan_labels = np.unique(clan_labels)

		clan_num_dict = {}
		for clan in unique_clan_labels:
			num_clan = 0
			for i in range(len(clan_labels)):
				if clan_labels[i] == clan:
					num_clan += 1
			clan_num_dict[clan] = num_clan
		
		min_clan_num = []
		for key in clan_num_dict:
			if len(min_clan_num) == 0:
				min_clan_num.append(clan_num_dict[key])
				min_clan = key
			elif clan_num_dict[key] < min_clan_num[-1]:
				min_clan_num[-1] = clan_num_dict[key]
				min_clan = key
		min_clan_num = min_clan_num[0]

		x_clan_data_imbal = ici_matrix[:,1:]
		x_clan_data = np.zeros((min_clan_num*2,x_clan_data_imbal.shape[1]))
		y_clan_data =[]

		num_clan_keys = 0
		for key in clan_num_dict:
			num_clan_keys += 1
			num_clan_index = 0
			for i in range(len(clan_labels)):
				if clan_labels[i]==key:
					if num_clan_index > min_clan_num-1:
						break
					else:
						x_clan_data[min_clan_num*(num_clan_keys-1)+num_clan_index,:] = \
							x_clan_data_imbal[i,:]
						y_clan_data.append(clan_labels[i])
						num_clan_index += 1

		x_clan_data = np.asarray(x_clan_data)
		x_clan_data = x_clan_data.reshape(x_clan_data.shape[0], x_clan_data.shape[1], 1)
		
		#y_clan_data = clan_labels
		unique_clans = unique_clan_labels
		
		class_mapping = {label:idx for idx,label in enumerate(unique_clans)}

		label_encoder = LabelEncoder()
		label_encoder.fit(y_clan_data)
		y_encoded_labels = label_encoder.transform(y_clan_data)
		y_one_hot = np_utils.to_categorical(y_encoded_labels)

		return class_mapping, unique_clans, x_clan_data, y_one_hot

	def whale_id_processing(self, removal_threshold=200):
		ici_matrix = self.gen_ici_matrix()
		whale_labels = self.read_file().iloc[:,self.whale_id_col].values
		unique_id_labels = np.unique(whale_labels)

		id_num_dict = {}
		for whale in unique_id_labels:
			num_whale = 0
			for i in range(len(whale_labels)):
				if whale_labels[i] == whale:
					num_whale += 1
			id_num_dict[whale] = num_whale
		
		x_whale_id_data = ici_matrix[:,1:]
		whale_removal = ['0','5151','5560','5586']
		for key in id_num_dict:
			if id_num_dict[key] < removal_threshold:
				whale_removal.append(key)

		for whale_rem in whale_removal:
			x_whale_id_data = x_whale_id_data[whale_labels!=whale_rem]
			whale_labels = whale_labels[whale_labels!=whale_rem]

		x_id_data = np.asarray(x_whale_id_data)
		x_id_data = x_id_data.reshape(x_id_data.shape[0], x_id_data.shape[1], 1)
		y_id_data = whale_labels
		
		#y_clan_data = clan_labels
		unique_ids = np.unique(whale_labels)
		
		class_mapping = {label:idx for idx,label in enumerate(unique_ids)}

		label_encoder = LabelEncoder()
		label_encoder.fit(y_id_data)
		y_encoded_labels = label_encoder.transform(y_id_data)
		y_one_hot = np_utils.to_categorical(y_encoded_labels)

		return class_mapping, unique_ids, x_id_data, y_one_hot
		
class ModelBuilding(PreProcessing):
	"""
	Parameters
	------------
	net1_lstm1_units : int
		Number of hidden units in the first LSTM layer of model1
	net1_lstm2_units : int
		Number of hidden units in the second LSTM layer of model1
	net1_acitvation : str
		Activation function used in constructing model1
	net1_optimizer : str
		Optimizer used in constructing model1
	net1_lr : float
		Learning rate during training of model1
	net1_epochs : int
		Number of epochs for model1 training
	netct_lstm1_units : int
		Number of hidden units in the first LSTM layer of modelct
	netct_dense1 : int
		Number of hidden units in the first debse layer of modelct
	netct_dense1_acitvation : str
		Activation function used in constructing modelct
	netct_dense2_activation : str
		Activation function used in constructing modelct
	net1_optimizer : str
		Optimizer used in constructing model1
	netct_lr : float
		Learning rate during training of modelct
	netct_epochs : int
		Number of epochs for modelct training

	Methods
	------------
	train_model1
		Builds the model trained on the sequence prediction task
	train_modelct
		Builds the model trained to classify codas based on coda type label
	train_model_clan
		Builds the model trained to classify codas based on vocal clan label
	train_model_id
		Builds the model trained to classify codas based on whale ID label
	show_model1_training
		Plot the MSE loss for model1
	show_model_ct_training
		Plot the training curves for the coda type network
	show_model_clan_training
		Plot the training curves for the clan type network
	show_model_id_training
		Plot the training curves for the whale ID network
	plot_model1
		Plots the output of the model1 prediciton

	"""

	def __init__(self, file_name, ici_start=4, ici_end=13,
		coda_type_col=13, clan_col=14, whale_id_col=17, pretrain_test_size=0.1):

		self.file_name = file_name
		PreProcessing.__init__(self, file_name, ici_start, ici_end,
		coda_type_col, clan_col, whale_id_col, pretrain_test_size)

	def train_model1(self,net1_lstm1_units=256, net1_lstm2_units=256, 
		net1_activation='sigmoid', net1_optimizer='adam',net1_lr=1e-3,
		net1_epochs=20):#20
		"""Pretrains the initial network"""

		x_train_net1, x_test_net1, y_train_net1, y_test_net1 = \
			PreProcessing(self.file_name).pretrain_processing()

		model1 = Sequential()
		model1.add(LSTM(net1_lstm1_units, return_sequences=True,
			input_shape=(x_train_net1.shape[1],1)))
		model1.add(LSTM(net1_lstm2_units, return_sequences=False))
		model1.add(Dense(1, activation=net1_activation))
		model1.compile(loss='mse', optimizer=net1_optimizer)

		print('Pretraining the model...')

		model1_fitting = model1.fit(x_train_net1, y_train_net1, epochs=net1_epochs,
			shuffle=False, verbose=1)
		model1.summary()
		metrics = model1.evaluate(x_train_net1, y_train_net1)
		print(metrics)

		y_test_matrix = np.zeros((y_test_net1.shape[0],2))
		y_test_matrix[:,0] = y_test_net1
		y_pred = model1.predict(x_test_net1, verbose = 0)
		for i in range(y_pred.size):
			y_test_matrix[i,1] = y_pred[i]

		err_vec = []
		for i in range(y_test_matrix.shape[0]):
			err = np.abs(y_test_matrix[i,1]-y_test_matrix[i,0])/y_test_matrix[i,0]
			err_vec.append(err)
		err_vec = np.asarray(err_vec)
		err_mean = err_vec.mean()
		print('Mean Relative Error =', err_mean)

		return model1

	def train_modelct(self, netct_lstm1_units=256, netct_dense1=512, 
		netct_dense1_activation = 'relu', netct_dense2_activation='softmax',
		netct_lr=1e-3, netct_epochs=25):#25
		"""Implements the transfer learning procedure to train
		a network to classify codas based on coda type"""

		class_mapping, unique_ct, x_ct_data, y_one_hot = \
			PreProcessing(self.file_name).coda_type_processing()
		num_classes = len(unique_ct)

		model1 = self.train_model1()

		for layer in model1.layers[:2]:
			layer.trainable = False

		layer_transfer = model1.layers[0].output
		layer_transfer = LSTM(netct_lstm1_units)(layer_transfer)
		layer_transfer = Dense(netct_dense1, activation=netct_dense1_activation)(layer_transfer)
		predictions = Dense(num_classes, activation=netct_dense2_activation)(layer_transfer)

		modelct = Model(input=model1.input, output=predictions)
		modelct.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(netct_lr),
			metrics=['accuracy'])

		print('Training the coda type model...')
		modelct_fitting=modelct.fit(x_ct_data, y_one_hot, epochs=netct_epochs, 
			shuffle=False, verbose=1)
		modelct.summary()
		metrics = modelct.evaluate(x_ct_data, y_one_hot)
		print(metrics)

		return modelct

	def train_model_clan(self, net_clan_lstm1_units=256, net_clan_dense1=256, 
		net_clan_dense1_activation = 'relu', net_clan_dense2_activation='softmax',
		net_clan_lr=1e-2, net_clan_epochs=15):
		"""Implements the transfer learning procedure to train
		a network to classify codas based on vocal clan"""

		class_mapping, unique_clans, x_clan_data, y_one_hot = \
			PreProcessing(self.file_name).vocal_clan_processing()
		num_classes = len(unique_clans)

		model1 = self.train_model1()

		for layer in model1.layers[:2]:
			layer.trainable = False

		layer_transfer = model1.layers[0].output
		layer_transfer = LSTM(net_clan_lstm1_units)(layer_transfer)
		layer_transfer = Dense(net_clan_dense1, 
			activation=net_clan_dense1_activation)(layer_transfer)
		predictions = Dense(num_classes, 
			activation=net_clan_dense2_activation)(layer_transfer)

		model_clan = Model(input=model1.input, output=predictions)
		model_clan.compile(loss='categorical_crossentropy', 
			optimizer=optimizers.Adam(net_clan_lr), metrics=['accuracy'])

		print('Training the vocal clan model...')
		model_clan_fitting=model_clan.fit(x_clan_data, y_one_hot, 
			epochs=net_clan_epochs, shuffle=True, verbose=1)
		model_clan.summary()
		metrics = model_clan.evaluate(x_clan_data, y_one_hot)
		print(metrics)
		return model_clan

	def train_model_id(self, net_id_lstm1_units=256, net_id_dense1=256, 
		net_id_dense1_activation = 'relu', net_id_dense2_activation='softmax',
		net_id_lr=1e-2, net_id_epochs=50):
		"""Implements the transfer learning procedure to train
		a network to classify codas based on whale ID"""

		class_mapping, unique_ids, x_id_data, y_one_hot = \
			PreProcessing(self.file_name).whale_id_processing()
		num_classes = len(unique_ids)

		model1 = self.train_model1()

		for layer in model1.layers[:2]:
			layer.trainable = False

		layer_transfer = model1.layers[0].output
		layer_transfer = LSTM(net_id_lstm1_units)(layer_transfer)
		layer_transfer = Dense(net_id_dense1, 
			activation=net_id_dense1_activation)(layer_transfer)
		predictions = Dense(num_classes, 
			activation=net_id_dense2_activation)(layer_transfer)

		model_id = Model(input=model1.input, output=predictions)
		model_id.compile(loss='categorical_crossentropy', 
			optimizer=optimizers.Adam(net_id_lr), metrics=['accuracy'])

		print('Training the vocal clan model...')
		model_id_fitting=model_id.fit(x_id_data, y_one_hot, 
			epochs=net_id_epochs, shuffle=True, verbose=1)
		model_id.summary()
		metrics = model_id.evaluate(x_id_data, y_one_hot)
		print(metrics)

		return model_id

	def show_model1_training(self,net1_lstm1_units=256, net1_lstm2_units=256, 
		net1_activation='sigmoid', net1_optimizer='adam',net1_lr=1e-3,
		net1_epochs=20):
		"""Plot the training loss for the foundation network"""
		
		x_train_net1, x_test_net1, y_train_net1, y_test_net1 = \
			PreProcessing(self.file_name).pretrain_processing()

		model1 = Sequential()
		model1.add(LSTM(net1_lstm1_units, return_sequences=True,
			input_shape=(x_train_net1.shape[1],1)))
		model1.add(LSTM(net1_lstm2_units, return_sequences=False))
		model1.add(Dense(1, activation=net1_activation))
		model1.compile(loss='mse', optimizer=net1_optimizer)

		print('Pretraining the model...')

		model1_fitting = model1.fit(x_train_net1, y_train_net1, epochs=net1_epochs,
			shuffle=False, verbose=1)
		model1.summary()

		x_int = [0, 5, 10, 15, 20]

		plt.figure()
		plt.plot(model1_fitting.history['loss'])
		plt.title('MSE Loss')
		plt.xticks(x_int)
		plt.ylabel('Loss')
		plt.xlabel('Epoch')
		#plt.legend(['Training'], loc='upper left')
		plt.show()
	
	def show_model_ct_training(self, netct_lstm1_units=256, netct_dense1=512, 
		netct_dense1_activation = 'relu', netct_dense2_activation='softmax',
		netct_lr=1e-3, netct_epochs=25):
		"""Show the training accuracy and loss curves for the coda type network"""
		
		class_mapping, unique_ct, x_ct_data, y_one_hot = \
			PreProcessing(self.file_name).coda_type_processing()
		num_classes = len(unique_ct)

		model1 = self.train_model1()

		for layer in model1.layers[:2]:
			layer.trainable = False

		layer_transfer = model1.layers[0].output
		layer_transfer = LSTM(netct_lstm1_units)(layer_transfer)
		layer_transfer = Dense(netct_dense1, activation=netct_dense1_activation)(layer_transfer)
		predictions = Dense(num_classes, activation=netct_dense2_activation)(layer_transfer)

		modelct = Model(input=model1.input, output=predictions)
		modelct.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(netct_lr),
			metrics=['accuracy'])

		print('Training the coda type model...')
		modelct_fitting=modelct.fit(x_ct_data, y_one_hot, epochs=netct_epochs, 
			shuffle=False, verbose=1)
		modelct.summary()
		
		plt.figure()
		plt.subplot(1,2,1)
		plt.plot(modelct_fitting.history['acc'])
		plt.title('Model Accuracy')
		plt.ylabel('Accuracy')
		plt.xlabel('Epoch')
		#plt.legend(['Training'], loc='upper left')
		plt.subplot(1,2,2)
		plt.plot(modelct_fitting.history['loss'])
		plt.title('Model Loss')
		plt.ylabel('Loss')
		plt.xlabel('Epoch')
		#plt.legend(['Training'], loc='upper left')
		#plt.savefig('ModelTraining.png')
		plt.show()

	def show_model_clan_training(self, net_clan_lstm1_units=256, net_clan_dense1=256, 
		net_clan_dense1_activation = 'relu', net_clan_dense2_activation='softmax',
		net_clan_lr=1e-2, net_clan_epochs=30):
		"""Show the training accuracy and loss curves for the clan type network"""
		
		class_mapping, unique_clans, x_clan_data, y_one_hot = \
			PreProcessing(self.file_name).vocal_clan_processing()
		num_classes = len(unique_clans)

		model1 = self.train_model1()

		for layer in model1.layers[:2]:
			layer.trainable = False

		layer_transfer = model1.layers[0].output
		layer_transfer = LSTM(net_clan_lstm1_units)(layer_transfer)
		layer_transfer = Dense(net_clan_dense1, 
			activation=net_clan_dense1_activation)(layer_transfer)
		predictions = Dense(num_classes, 
			activation=net_clan_dense2_activation)(layer_transfer)

		model_clan = Model(input=model1.input, output=predictions)
		model_clan.compile(loss='categorical_crossentropy', 
			optimizer=optimizers.Adam(net_clan_lr), metrics=['accuracy'])

		print('Training the vocal clan model...')
		model_clan_fitting=model_clan.fit(x_clan_data, y_one_hot, 
			epochs=net_clan_epochs, shuffle=True, verbose=1)
		model_clan.summary()

		plt.figure()
		plt.subplot(1,2,1)
		plt.plot(model_clan_fitting.history['acc'])
		plt.title('Model Accuracy')
		plt.ylabel('Accuracy')
		plt.xlabel('Epoch')
		#plt.legend(['Training'], loc='upper left')
		plt.subplot(1,2,2)
		plt.plot(model_clan_fitting.history['loss'])
		plt.title('Model Loss')
		plt.ylabel('Loss')
		plt.xlabel('Epoch')
		#plt.legend(['Training'], loc='upper left')
		#plt.savefig('ModelTraining.png')
		plt.show()

	def show_model_id_training(self, net_id_lstm1_units=256, net_id_dense1=256, 
		net_id_dense1_activation = 'relu', net_id_dense2_activation='softmax',
		net_id_lr=1e-2, net_id_epochs=50):
		"""Show the training accuracy and loss curves for the whale ID network"""
		
		class_mapping, unique_ids, x_id_data, y_one_hot = \
			PreProcessing(self.file_name).whale_id_processing()
		num_classes = len(unique_ids)

		model1 = self.train_model1()

		for layer in model1.layers[:2]:
			layer.trainable = False

		layer_transfer = model1.layers[0].output
		layer_transfer = LSTM(net_id_lstm1_units)(layer_transfer)
		layer_transfer = Dense(net_id_dense1, 
			activation=net_id_dense1_activation)(layer_transfer)
		predictions = Dense(num_classes, 
			activation=net_id_dense2_activation)(layer_transfer)

		model_id = Model(input=model1.input, output=predictions)
		model_id.compile(loss='categorical_crossentropy', 
			optimizer=optimizers.Adam(net_id_lr), metrics=['accuracy'])

		print('Training the vocal clan model...')
		model_id_fitting=model_id.fit(x_id_data, y_one_hot, 
			epochs=net_id_epochs, shuffle=True, verbose=1)
		model_id.summary()

		plt.figure()
		plt.subplot(1,2,1)
		plt.plot(model_id_fitting.history['acc'])
		plt.title('Model Accuracy')
		plt.ylabel('Accuracy')
		plt.xlabel('Epoch')
		#plt.legend(['Training'], loc='upper left')
		plt.subplot(1,2,2)
		plt.plot(model_id_fitting.history['loss'])
		plt.title('Model Loss')
		plt.ylabel('Loss')
		plt.xlabel('Epoch')
		#plt.legend(['Training'], loc='upper left')
		#plt.savefig('ModelTraining.png')
		plt.show()

	def plot_model1(self,net1_lstm1_units=256, net1_lstm2_units=256, 
		net1_activation='sigmoid', net1_optimizer='adam',net1_lr=1e-3,
		net1_epochs=20):
		"""Plot the model1 prediction"""

		x_train_net1, x_test_net1, y_train_net1, y_test_net1 = \
			PreProcessing(self.file_name).pretrain_processing()

		model1 = Sequential()
		model1.add(LSTM(net1_lstm1_units, return_sequences=True,
			input_shape=(x_train_net1.shape[1],1)))
		model1.add(LSTM(net1_lstm2_units, return_sequences=False))
		model1.add(Dense(1, activation=net1_activation))
		model1.compile(loss='mse', optimizer=net1_optimizer)

		print('Pretraining the model...')

		model1_fitting = model1.fit(x_train_net1, y_train_net1, epochs=net1_epochs,
			shuffle=False, verbose=1)
		model1.summary()
		metrics = model1.evaluate(x_train_net1, y_train_net1)
		print(metrics)

		y_test_matrix = np.zeros((y_test_net1.shape[0],2))
		y_test_matrix[:,0] = y_test_net1
		y_pred = model1.predict(x_test_net1, verbose = 0)
		for i in range(y_pred.size):
			y_test_matrix[i,1] = y_pred[i]

		err_vec = []
		for i in range(y_test_matrix.shape[0]):
			err = np.abs(y_test_matrix[i,1]-y_test_matrix[i,0])/y_test_matrix[i,0]
			err_vec.append(err)
		err_vec = np.asarray(err_vec)
		err_mean = err_vec.mean()
		print('Mean Relative Error =', err_mean)

		x_test_vec1 = x_test_net1[0,:,:]
		x_test_vec1 = x_test_vec1.reshape(x_test_vec1.shape[1],x_test_vec1.shape[0])
		
		y_test_1 = y_test_net1[0]
		x_true = np.append(x_test_vec1, y_test_1)
		y_pred_1 = y_pred[0]
		x_pred = np.append(x_test_vec1, y_pred_1)

		x_true = np.cumsum(x_true)
		x_pred = np.cumsum(x_pred)

		x_axis = np.arange(1,len(x_true)+1)

		plt.figure()
		plt.plot(x_axis, x_pred, marker='o',color='b')
		plt.plot(x_axis, x_true, marker='o',color='r')
		plt.legend(('Predicted Value', 'True Value'))
		plt.xlabel('Click')
		plt.ylabel('Cumulative Coda Duration (s)')
		plt.title('Pretrained Model Output')
		plt.show()

		return model1

class SaveModel(ModelBuilding):
	"""Save the model

	Parameters
	------------
	model1_vis : bool
		Set to true to visualize model1
	modelct_vis : bool
		Set to true to visualize modelct
	model_clan_vis : bool
		Set to true to visualize model_clan
	model_id_vis : bool
		Set to true to visualize model_id

	Methods
	------------
	model_ct_save
		Save the coda type model
	model_clan_save
		Save the clan model
	model_id_save
		Save the ID model
	"""

	def __init__(self, file_name, ici_start=4, ici_end=13,
		coda_type_col=13, clan_col=14, whale_id_col=17, pretrain_test_size=0.1,
		model1_vis=False, modelct_vis=False, model_clan_vis=False, 
		model_id_vis=False):

		self.file_name = file_name
		ModelBuilding.__init__(self, file_name)

		self.modelct_vis = modelct_vis
		self.model_clan_vis = model_clan_vis
		self.model_id_vis = model_id_vis
		
		if model1_vis:
			self.model1 = ModelBuilding(file_name).train_model1()
		if modelct_vis:
			self.modelct = ModelBuilding(file_name).train_modelct()
		if model_clan_vis:
			self.model_clan = ModelBuilding(file_name).train_model_clan()
		if model_id_vis:
			self.model_id = ModelBuilding(file_name).train_model_id()

	def model_ct_save(self):
		"""Save the coda type model"""

		if self.modelct_vis:
			self.modelct.save('CodaTypeModel.h5')
		else:
			print('Error saving the coda type model')

	def model_clan_save(self):
		"""Save the clan model"""

		if self.model_clan_vis:
			self.model_clan.save('ClanModel.h5')
		else:
			print('Error saving the clan model')

	def model_id_save(self):
		"""Save the ID model"""

		if self.model_id_vis:
			self.model_id.save('IDModel.h5')
		else:
			print('Error saving the ID model')

class ModelVisualization(ModelBuilding):
	"""Use PCA and t-SNE to visualize the extracted features
	
	Parameters
	------------
	model1_vis : bool
		Set to true to visualize model1
	modelct_vis : bool
		Set to true to visualize modelct
	model_clan_vis : bool
		Set to true to visualize model_clan
	model_id_vis : bool
		Set to true to visualize model_id


	Methods
	------------
	get_layers_model1
		Print the architecture of model1
	get_layers_modelct
		Print the arctitecture of modelct
	get_layers_model_clan
		Print the arctitecture of model_clan
	get_layers_model_id
		Print the arctitecture of model_id
	get_model1_architecture
		Plot and save a model of the architecture
	get_model_ct_architecture
		Plot and save a model of the architecture
	get_model_clan_architecture
		Plot and save a model of the architecture
	get_model_id_architecture
		Plot and save a model of the architecture
	gen_trunc_model1
		Generate a truncated model1
	gen_trunc_modelct
		Generate a truncated modelct
	gen_trunc_model_clan
		Generate a truncated model_clan
	gen_trunc_model_id
		Generate a truncated model_id
	tsne_visualization_modelct
		Plot the tSNE visualization
	tsne_visualization_modelct3D
		Plot the 3D tSNE visualization
	tsne_visualization_model_clan
		Plot the tSNE visualization
	tsne_visualization_model_id
		Plot the tSNE visualization
	get_PCA_variance_ct
		Shows the explained PCA variance for coda type
	get_PCA_variance_clan
		Shows the explained PCA variance for clan class
	get_PCA_variance_id
		Shows the explained PCA variance for whale ID
	"""
	
	def __init__(self, file_name, ici_start=4, ici_end=13,
		coda_type_col=13, clan_col=14, whale_id_col=17, pretrain_test_size=0.1,
		model1_vis=False, modelct_vis=False, model_clan_vis=False, 
		model_id_vis=False):

		self.file_name = file_name
		ModelBuilding.__init__(self, file_name)
		
		if model1_vis:
			self.model1 = ModelBuilding(file_name).train_model1()
		if modelct_vis:
			self.modelct = ModelBuilding(file_name).train_modelct()
		if model_clan_vis:
			self.model_clan = ModelBuilding(file_name).train_model_clan()
		if model_id_vis:
			self.model_id = ModelBuilding(file_name).train_model_id()

		self.x_train_net1, self.x_test_net1, self.y_train_net1, self.y_test_net1 = \
			PreProcessing(self.file_name).pretrain_processing()

		self.class_mapping, self.unique_ct, self.x_ct_data, self.y_one_hot = \
			PreProcessing(self.file_name).coda_type_processing()
		self.num_ct_classes = len(self.unique_ct)

		self.class_mapping_clan, self.unique_clan, self.x_clan_data, self.y_one_hot_clan = \
			PreProcessing(self.file_name).vocal_clan_processing()
		self.num_clan_classes = len(self.unique_clan)

		self.class_mapping_id, self.unique_id, self.x_id_data, self.y_one_hot_id = \
			PreProcessing(self.file_name).whale_id_processing()
		self.num_id_classes = len(self.unique_id)

	def get_layers_model1(self):
		"""Print the architecture"""

		for layer in self.model1.layers:
			print(layer.name, layer.trainable)
			print('Layer Configuration:')
			print(layer.get_config(), end='\n{}\n'.format('----'*10))

	def get_layers_modelct(self):
		"""Print the architecture"""

		for layer in self.modelct.layers:
			print(layer.name, layer.trainable)
			print('Layer Configuration:')
			print(layer.get_config(), end='\n{}\n'.format('----'*10))

	def get_layers_model_clan(self):
		"""Print the architecture"""

		for layer in self.model_clan.layers:
			print(layer.name, layer.trainable)
			print('Layer Configuration:')
			print(layer.get_config(), end='\n{}\n'.format('----'*10))

	def get_layers_model_id(self):
		"""Print the architecture"""

		for layer in self.model_id.layers:
			print(layer.name, layer.trainable)
			print('Layer Configuration:')
			print(layer.get_config(), end='\n{}\n'.format('----'*10))

	def get_model1_architecture(self):
		"""Plot and save a model of the architecture"""

		from keras.utils import plot_model
		model1 = ModelBuilding(self.file_name).train_model1()

		try:
			plot_model(model1, to_file='Model1.png')
		except FileExistsError:
			print('File already exists')

	def get_model_ct_architecture(self):
		"""Plot and save a model of the architecture"""

		from keras.utils import plot_model
		try:
			plot_model(self.modelct, to_file='Model_CT.png')
		except FileExistsError:
			print('File already exists')

	def get_model_clan_architecture(self):
		"""Plot and save a model of the architecture"""

		from keras.utils import plot_model
		try:
			plot_model(self.model_clan, to_file='Model_Clan.png')
		except FileExistsError:
			print('File already exists')

	def get_model_id_architecture(self):
		"""Plot and save a model of the architecture"""

		from keras.utils import plot_model
		try:
			plot_model(self.model_id, to_file='Model_ID.png')
		except FileExistsError:
			print('File already exists')

	def gen_trunc_model1(self):
		"""Build a truncated model"""

		trained_model = self.model1
		net1_lstm1_units = trained_model.layers[0].units
		net1_lstm2_units = trained_model.layers[1].units
		net1_activation = trained_model.layers[2].activation
		net1_optimizer = trained_model.optimizer

		trunc_model = Sequential()
		trunc_model.add(LSTM(net1_lstm1_units, return_sequences=True,
			input_shape=(self.x_train_net1.shape[1],1)))
		trunc_model.add(LSTM(net1_lstm2_units, return_sequences=False))
		#trunc_model.add(Dense(1, activation=net1_activation))
		for i, layer in enumerate(trunc_model.layers):
			layer.set_weights(trained_model.layers[i].get_weights())
		trunc_model.compile(loss='mse', optimizer=net1_optimizer)
		return trunc_model
	
	def gen_trunc_modelct(self):
		"""Build a truncated model"""

		trained_model = self.modelct
		lstm_units1 = trained_model.layers[1].units
		lstm_units2 = trained_model.layers[2].units
		trained_model_optimizer=trained_model.optimizer

		trunc_model = Sequential()
		trunc_model.add(LSTM(lstm_units1, return_sequences=True,
			input_shape=(self.x_train_net1.shape[1],1)))
		trunc_model.add(LSTM(lstm_units2, return_sequences=False))
		for i, layer in enumerate(trunc_model.layers):
			layer.set_weights(trained_model.layers[i+1].get_weights())
		trunc_model.compile(loss='mse', optimizer=trained_model_optimizer)
		return trunc_model

	def gen_trunc_model_clan(self):
		"""Build a truncated model"""

		trained_model = self.model_clan
		lstm_units1 = trained_model.layers[1].units
		lstm_units2 = trained_model.layers[2].units
		trained_model_optimizer=trained_model.optimizer

		trunc_model = Sequential()
		trunc_model.add(LSTM(lstm_units1, return_sequences=True,
			input_shape=(self.x_train_net1.shape[1],1)))
		trunc_model.add(LSTM(lstm_units2, return_sequences=False))
		for i, layer in enumerate(trunc_model.layers):
			layer.set_weights(trained_model.layers[i+1].get_weights())
		trunc_model.compile(loss='mse', optimizer=trained_model_optimizer)
		return trunc_model

	def gen_trunc_model_id(self):
		"""Build a truncated model"""

		trained_model = self.model_id
		lstm_units1 = trained_model.layers[1].units
		lstm_units2 = trained_model.layers[2].units
		trained_model_optimizer=trained_model.optimizer

		trunc_model = Sequential()
		trunc_model.add(LSTM(lstm_units1, return_sequences=True,
			input_shape=(self.x_train_net1.shape[1],1)))
		trunc_model.add(LSTM(lstm_units2, return_sequences=False))
		for i, layer in enumerate(trunc_model.layers):
			layer.set_weights(trained_model.layers[i+1].get_weights())
		trunc_model.compile(loss='mse', optimizer=trained_model_optimizer)
		return trunc_model

	def tsne_visualization_modelct(self):
		"""Visualize the activations"""

		trunc_model = self.gen_trunc_modelct()
		extracted_features = trunc_model.predict(self.x_ct_data)
		
		from sklearn.decomposition import PCA
		from sklearn.manifold import TSNE

		pca = PCA(n_components=20)
		pca_result = pca.fit_transform(extracted_features)
		print('Variance PCA: {}'.format(np.sum(pca.explained_variance_ratio_)))

		tsne = TSNE(n_components=2, verbose = 1)
		tsne_results = tsne.fit_transform(pca_result)

		convert_one_hot = [np.where(r==1)[0][0] for r in self.y_one_hot]

		y_test_cat = np_utils.to_categorical(convert_one_hot, num_classes = self.num_ct_classes)
		color_map = np.argmax(y_test_cat, axis=1)
		colors = cm.rainbow(np.linspace(0, 1, self.num_ct_classes))

		plt.figure(figsize=(10,10))
		for cl in range(self.num_ct_classes):
			indices = np.where(color_map==cl)
			indices = indices[0]
			c = colors[cl]
			plt.scatter(tsne_results[indices,0], tsne_results[indices, 1], 
				label=cl, color=c)
		print('Class Mapping:', self.class_mapping)
		plt.legend()
		plt.xlabel('Axis 1')
		plt.ylabel('Axis 2')
		plt.title('t-SNE Visualization')
		plt.show()

	def tsne_visualization_modelct3D(self):
		"""Visualize the activations"""

		trunc_model = self.gen_trunc_modelct()
		extracted_features = trunc_model.predict(self.x_ct_data)
		
		from sklearn.decomposition import PCA
		from sklearn.manifold import TSNE

		pca = PCA(n_components=20)
		pca_result = pca.fit_transform(extracted_features)
		print('Variance PCA: {}'.format(np.sum(pca.explained_variance_ratio_)))

		tsne = TSNE(n_components=3, verbose = 1)
		tsne_results = tsne.fit_transform(pca_result)

		convert_one_hot = [np.where(r==1)[0][0] for r in self.y_one_hot]

		y_test_cat = np_utils.to_categorical(convert_one_hot, num_classes = self.num_ct_classes)
		color_map = np.argmax(y_test_cat, axis=1)
		colors = cm.rainbow(np.linspace(0, 1, self.num_ct_classes))

		fig = plt.figure(figsize=(10,10))
		ax = fig.add_subplot(111, projection='3d')
		for cl in range(self.num_ct_classes):
			indices = np.where(color_map==cl)
			indices = indices[0]
			c = colors[cl]
			ax.scatter(tsne_results[indices,0], tsne_results[indices, 1], tsne_results[indices, 2], 
				label=cl, color=c)
		print('Class Mapping:', self.class_mapping)
		#ax.legend()
		ax.set_xlabel('Axis 1')
		ax.set_ylabel('Axis 2')
		ax.set_zlabel('Axis 3')
		ax.set_title('t-SNE Visualization')
		plt.show()

	def tsne_visualization_model_clan(self):
		"""Visualize the activations"""

		trunc_model = self.gen_trunc_model_clan()
		extracted_features = trunc_model.predict(self.x_clan_data)
		
		from sklearn.decomposition import PCA
		from sklearn.manifold import TSNE

		pca = PCA(n_components=20)
		pca_result = pca.fit_transform(extracted_features)
		print('Variance PCA: {}'.format(np.sum(pca.explained_variance_ratio_)))

		tsne = TSNE(n_components=2, verbose = 1)
		tsne_results = tsne.fit_transform(pca_result)

		convert_one_hot = [np.where(r==1)[0][0] for r in self.y_one_hot_clan]
		y_test_cat = np_utils.to_categorical(convert_one_hot, num_classes = self.num_clan_classes)
		color_map = np.argmax(y_test_cat, axis=1)
		colors = cm.rainbow(np.linspace(0, 1, self.num_clan_classes))

		plt.figure(figsize=(10,10))
		for cl in range(self.num_clan_classes):
			indices = np.where(color_map==cl)
			indices = indices[0]
			c = colors[cl]
			plt.scatter(tsne_results[indices,0], tsne_results[indices, 1], 
				label=cl, color=c)
		print('Class Mapping:', self.class_mapping_clan)
		plt.legend()
		plt.xlabel('Axis 1')
		plt.ylabel('Axis 2')
		plt.title('t-SNE Visualization')
		plt.show()

	def tsne_visualization_model_id(self):
		"""Visualize the activations"""

		trunc_model = self.gen_trunc_model_id()
		extracted_features = trunc_model.predict(self.x_id_data)
		
		from sklearn.decomposition import PCA
		from sklearn.manifold import TSNE

		pca = PCA(n_components=20)
		pca_result = pca.fit_transform(extracted_features)
		print('Variance PCA: {}'.format(np.sum(pca.explained_variance_ratio_)))

		tsne = TSNE(n_components=2, verbose = 1)
		tsne_results = tsne.fit_transform(pca_result)

		convert_one_hot = [np.where(r==1)[0][0] for r in self.y_one_hot_id]
		y_test_cat = np_utils.to_categorical(convert_one_hot, num_classes = self.num_id_classes)
		color_map = np.argmax(y_test_cat, axis=1)
		colors = cm.rainbow(np.linspace(0, 1, self.num_id_classes))

		plt.figure(figsize=(10,10))
		for cl in range(self.num_id_classes):
			indices = np.where(color_map==cl)
			indices = indices[0]
			c = colors[cl]
			plt.scatter(tsne_results[indices,0], tsne_results[indices, 1], 
				label=cl, color=c)
		print('Class Mapping:', self.class_mapping_id)
		plt.legend()
		plt.xlabel('Axis 1')
		plt.ylabel('Axis 2')
		plt.title('t-SNE Visualization')
		plt.show()

	def get_PCA_variance_ct(self):
		"""Shows the explained PCA variance ratios"""

		trunc_model = self.gen_trunc_modelct()
		extracted_features = trunc_model.predict(self.x_ct_data)
		
		from sklearn.decomposition import PCA
		from sklearn.manifold import TSNE

		for component in [255, 254, 253, 200, 100, 50, 25, 15, 10, 7, 6, 5, 4, 3, 2, 1]:
			pca = PCA(n_components=component)
			pca_result = pca.fit_transform(extracted_features)
			print('Number of Dimensions:',component)
			print('Variance PCA: {}'.format(np.sum(pca.explained_variance_ratio_)))

	def get_PCA_variance_clan(self):
		"""Shows the explained PCA variance ratios"""

		trunc_model = self.gen_trunc_model_clan()
		extracted_features = trunc_model.predict(self.x_clan_data)
		
		from sklearn.decomposition import PCA
		from sklearn.manifold import TSNE

		for component in [255, 254, 253, 200, 100, 50, 25, 15, 10, 7, 6, 5, 4, 3, 2, 1]:
			pca = PCA(n_components=component)
			pca_result = pca.fit_transform(extracted_features)
			print('Number of Dimensions:',component)
			print('Variance PCA: {}'.format(np.sum(pca.explained_variance_ratio_)))

	def get_PCA_variance_id(self):
		"""Shows the explained PCA variance ratios"""

		trunc_model = self.gen_trunc_model_id()
		extracted_features = trunc_model.predict(self.x_id_data)
		
		from sklearn.decomposition import PCA
		from sklearn.manifold import TSNE

		for component in [255, 254, 253, 200, 100, 50, 25, 15, 10, 7, 6, 5, 4, 3, 2, 1]:
			pca = PCA(n_components=component)
			pca_result = pca.fit_transform(extracted_features)
			print('Number of Dimensions:',component)
			print('Variance PCA: {}'.format(np.sum(pca.explained_variance_ratio_)))

class ModelTesting():
	"""Test the models
	
	Parameters
	------------
	file : str
		Name of the coda spreadsheet csv
	csv_file :
		Name of the csv file to save the prediction-test comparison matrix
	loaded model : str
		Name of the saved .h5 model

	Methods
	------------
	coda_type_testing
		Test the coda type model, print the accuracy, save the csv file

	"""

	def __init__(self, file='GeroDominicaCodasCSV.csv',csv_file='CodaTypeTesting.csv', 
		loaded_model='CodaTypeModel.h5'):
		self.file = file
		self.csv_file = csv_file
		self.loaded_model = loaded_model

	def coda_type_testing(self):
		"""Test the coda type predictions"""

		coda_type_preprocess = PreProcessing(self.file)
		ct_class_mapping, unique_ct, x_ct_data, y_one_hot_ct = \
			coda_type_preprocess.coda_type_processing()

		reverse_ct_class_mapping = {}
		for key, item in ct_class_mapping.items():
			reverse_ct_class_mapping[item] = key

		y_label_str = []
		for i in range(y_one_hot_ct.shape[0]):
			for j in range(y_one_hot_ct.shape[1]):
				if y_one_hot_ct[i,j] == 1:
					y_label_str.append(reverse_ct_class_mapping[j])

		loaded_ct_model = load_model(self.loaded_model)
		y_pred = loaded_ct_model.predict(x_ct_data)
		y_pred = np.argmax(y_pred,axis=1)

		y_pred_str = []
		for i in range(y_pred.shape[0]):
			y_pred_str.append(reverse_ct_class_mapping[y_pred[i]])

		num_eq = 0
		num_tot = len(y_label_str)
		for i in range(num_tot):
			if y_label_str[i] == y_pred_str[i]:
				num_eq += 1

		acc_str = 'Accuracy: ' + str(np.around(100*num_eq/num_tot, decimals=2)) + '%'

		print(acc_str)

		data_dict = {'Test': y_label_str, 'Pred': y_pred_str}
		ct_testing_df = pd.DataFrame(data=data_dict)
		ct_testing_df.to_csv(self.csv_file, encoding='utf-8', index=False)

"""
Examples
--------------------------------------------------
"""

#model_construction = ModelBuilding('GeroDominicaCodasCSV.csv')
#model1_net = model_construction.train_model1()

#model_construction = ModelBuilding('GeroDominicaCodasCSV.csv')
#ct_model = model_construction.train_model_ct()

#model_construction = ModelBuilding('GeroDominicaCodasCSV.csv')
#clan_model = model_construction.train_model_clan()

#model_construction = ModelBuilding('GeroDominicaCodasCSV.csv')
#id_model = model_construction.train_model_id()

#model_ct = ModelVisualization('GeroDominicaCodasCSV.csv', modelct_vis=True)
#model_ct.tsne_visualization_modelct3D()

#model_clan = ModelVisualization('GeroDominicaCodasCSV.csv', model_clan_vis=True)
#model_clan.tsne_visualization_model_clan()

#model_id = ModelVisualization('GeroDominicaCodasCSV.csv', model_id_vis=True)
#model_id.tsne_visualization_model_id()

#model_ct = ModelVisualization('GeroDominicaCodasCSV.csv', modelct_vis=True)
#model_ct.get_PCA_variance_ct()

#save_ct_model = SaveModel('GeroDominicaCodasCSV.csv', modelct_vis=True)
#save_ct_model.model_ct_save()

#model_ct_plot = ModelVisualization('GeroDominicaCodasCSV.csv', modelct_vis=True)
#model_ct_plot.get_model_ct_architecture()

#test_model = ModelTesting()
#test_model.coda_type_testing()

#model1 = ModelBuilding('GeroDominicaCodasCSV.csv')
#model1.plot_model1()

#model1 = ModelBuilding('GeroDominicaCodasCSV.csv')
#model1.show_model1_training()

"""
Run Code Below
--------------------------------------------------
"""

#model_clan = ModelVisualization('GeroDominicaCodasCSV.csv', model_clan_vis=True)
#model_clan.tsne_visualization_model_clan()

model = ModelBuilding('GeroDominicaCodasCSV.csv')
model.show_model_id_training()

