import cv2
import numpy as np
import os
import random
from numpy.random import seed
seed(1)
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.utils import plot_model
from tensorflow import set_random_seed
set_random_seed(1)

training_data = 'Training Folder'
testing_data = 'Testing Folder'
false_pos_data = 'False Positive Folder'

class Label(object):
	"""Create the labels for image data
	
	Parameters
	------------
	image_file : str
		Name of the image file
	
	Methods:
	------------
	gen_label
		Generates the label for the image
	one_hot_label
		Applies the label to the image
	"""

	def __init__(self, image_file):
		self.image_file = image_file

	def gen_label(self):
		"""Generate the label
		Assumes the file name is of the form
		'Click#.png' or 'NoClick#.png'"""

		label = ''
		for c in self.image_file:
			try:
				int(c)
				break
			except ValueError:
				label += c
		return label

	def one_hot_label(self):
		"""Apply the label to the image file"""

		label = Label(self.image_file).gen_label()
		if label == 'Click':
			o_h_label = np.array([1,0])
		else:
			o_h_label = np.array([0,1])
		return o_h_label

def label_training_data():
	"""Label the training data"""

	training_images = []
	for i in tqdm(os.listdir(training_data)):
		path = os.path.join(training_data, i)
		image_file = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
		image_file = cv2.resize(image_file, (64,64))
		training_images.append([np.array(image_file),
			Label(i).one_hot_label()])
	random.seed(1)
	random.shuffle(training_images)
	return training_images

def label_testing_data():
	"""Label the testing data"""

	testing_images = []
	for i in tqdm(os.listdir(testing_data)):
		path = os.path.join(testing_data, i)
		image_file = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
		image_file = cv2.resize(image_file, (64,64))
		testing_images.append([np.array(image_file),
			Label(i).one_hot_label()])
	return testing_images

def label_false_pos_data():
	"""Label the false positive data"""
	no_click_images = []
	for i in tqdm(os.listdir(false_pos_data)):
		path = os.path.join(false_pos_data, i)
		image_file = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
		image_file = cv2.resize(image_file, (64,64))
		no_click_images.append([np.array(image_file),
			Label(i).one_hot_label()])
	return no_click_images

from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *

training_images = label_training_data()
testing_images = label_testing_data()
false_pos_images = label_false_pos_data()

train_image_data = np.array([i[0] for i in
	training_images]).reshape(-1,64,64,1)
train_label_data = np.array([i[1] for i in
	training_images])

test_image_data = np.array([i[0] for i in
	testing_images]).reshape(-1,64,64,1)
test_label_data = np.array([i[1] for i in
	testing_images])

false_pos_image_data = np.array([i[0] for i in
	false_pos_images]).reshape(-1,64,64,1)
false_pos_label_data = np.array([i[1] for i in
	false_pos_images])

learning_rates = [1e-3]#[1e-4,1e-3]#[1e-4,1e-3,1e-2,1e-1]
batch_sizes = [64]#[32,64]#[64, 128]
filter_sizes = [5]#[3,5]#[3, 5]
activation_fxs = ['sigmoid']#['relu','tanh','sigmoid']#['relu','sigmoid']
accuracy = 0
accuracy_vec = []
for activation_fx in activation_fxs:
	for batch_size_n in batch_sizes:
		for learning_rate in learning_rates:
			for filter_size in filter_sizes:
				model = Sequential()

				model.add(InputLayer(input_shape = [64,64,1]))
				model.add(Conv2D(filters = 32, kernel_size = filter_size, strides = 1, 
					padding = 'same', activation = activation_fx))
				model.add(MaxPool2D(pool_size = 5, padding = 'same'))

				model.add(Conv2D(filters = 50, kernel_size = filter_size, strides = 1, 
					padding = 'same', activation = activation_fx))
				model.add(MaxPool2D(pool_size = 5, padding = 'same'))

				model.add(Conv2D(filters = 80, kernel_size = filter_size, strides = 1, 
					padding = 'same', activation = activation_fx))
				model.add(MaxPool2D(pool_size = 5, padding = 'same'))

				model.add(Dropout(0.25))
				model.add(Flatten())
				model.add(Dense(512, activation = activation_fx))
				model.add(Dropout(rate = 0.5))
				model.add(Dense(2, activation = 'softmax'))
				optimizer = Adam(lr = learning_rate)

				model.compile(optimizer = optimizer, 
					loss = 'categorical_crossentropy', metrics = ['accuracy'])
				model_fitting = model.fit(x = train_image_data, y = train_label_data, 
					epochs = 50, batch_size = batch_size_n, shuffle = False)
				model.summary()
				metrics = model.evaluate(x = train_image_data, y = train_label_data)

				loss = metrics[0]
				accuracy = metrics[1]
				accuracy_vec.append(accuracy)

				if len(accuracy_vec) == 1:
					hyperparameter_dict = {'Learning Rate' : learning_rate, 
						'Batch Size' : batch_size_n, 'Filter Size' : filter_size,
						'Activation Function' : activation_fx}
					optimal_accuracy = accuracy
				else:
					if accuracy_vec[-1] > accuracy_vec[-2]:
						hyperparameter_dict = {'Learning Rate' : learning_rate, 
							'Batch Size' : batch_size_n, 'Filter Size' : filter_size, 
							'Activation Function' : activation_fx}
						optimal_accuracy = accuracy_vec[-1]
print('Optimal choice of hyperparameters: ', hyperparameter_dict)
print('Accuracy: ', optimal_accuracy)

plt.figure()
plt.subplot(1,2,1)
plt.plot(model_fitting.history['acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
#plt.legend(['Training'], loc='upper left')
plt.subplot(1,2,2)
plt.plot(model_fitting.history['loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
#plt.legend(['Training'], loc='upper left')
#plt.savefig('ModelTraining.png')
plt.show()

try:
	plot_model(model, to_file='model.png')
except FileExistsError:
	print('File already exists')

fig = plt.figure(figsize = (14,14))
#for cnt, data in enumerate(testing_images[10:40]):
	#y = fig.add_subplot(6, 5, cnt+1)
for cnt, data in enumerate(testing_images[22:24]):#[10:40]):
	y = fig.add_subplot(1, 2, cnt+1)
	img = data[0]
	data = img.reshape(1,64,64,1)
	model_out = model.predict([data])

	if np.argmax(model_out) == 1:
		str_label = 'No Click'
	else:
		str_label = 'Click'

	y.imshow(img, cmap = 'gray')
	plt.title(str_label)
	y.axes.get_xaxis().set_visible(False)
	y.axes.get_yaxis().set_visible(False)
plt.show()	

figFP = plt.figure(figsize = (14,14))
for cnt, data in enumerate(false_pos_images[10:40]):
	y = figFP.add_subplot(6, 5, cnt+1)
	img = data[0]
	data = img.reshape(1,64,64,1)
	model_out = model.predict([data])

	if np.argmax(model_out) == 1:
		str_label = 'No Click'
	else:
		str_label = 'Click'

	y.imshow(img, cmap = 'gray')
	plt.title(str_label)
	y.axes.get_xaxis().set_visible(False)
	y.axes.get_yaxis().set_visible(False)
plt.show(figFP)	

opt_lr = hyperparameter_dict['Learning Rate']
opt_bs = hyperparameter_dict['Batch Size']
opt_filt_size = hyperparameter_dict['Filter Size']
opt_act_fx = hyperparameter_dict['Activation Function']


opt_model = Sequential()

opt_model.add(InputLayer(input_shape = [64,64,1]))
opt_model.add(Conv2D(filters = 32, kernel_size = opt_filt_size, strides = 1, 
	padding = 'same', activation = opt_act_fx))
opt_model.add(MaxPool2D(pool_size = 5, padding = 'same'))

opt_model.add(Conv2D(filters = 50, kernel_size = opt_filt_size, strides = 1, 
	padding = 'same', activation = opt_act_fx))
opt_model.add(MaxPool2D(pool_size = 5, padding = 'same'))

opt_model.add(Conv2D(filters = 80, kernel_size = opt_filt_size, strides = 1, 
	padding = 'same', activation = opt_act_fx))
opt_model.add(MaxPool2D(pool_size = 5, padding = 'same'))

opt_model.add(Dropout(0.25))
opt_model.add(Flatten())
opt_model.add(Dense(512, activation = opt_act_fx))
opt_model.add(Dropout(rate = 0.5))
opt_model.add(Dense(2, activation = 'softmax'))
opt_optimizer = Adam(lr = opt_lr)

opt_model.compile(optimizer = opt_optimizer, 
	loss = 'categorical_crossentropy', metrics = ['accuracy'])
opt_model_fitting = opt_model.fit(x = train_image_data, y = train_label_data, 
	epochs = 50, batch_size = opt_bs, shuffle = False)
opt_model.summary()

metrics_test = opt_model.evaluate(x = test_image_data, y = test_label_data)
accuracy_test = metrics_test[1]
print('Number of test images: ', len(test_label_data))
print('Test Accuracy: ', accuracy_test)

metrics_FA = opt_model.evaluate(x = false_pos_image_data, y = false_pos_label_data)
accuracy_FA = metrics_FA[1]
print('Number of false alarm images: ', len(false_pos_label_data))
print('False Alarm Accuracy: ', accuracy_FA)

plt.figure
plt.subplot(1,2,1)
plt.plot(opt_model_fitting.history['acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
#plt.legend(['Training'], loc='upper left')
plt.subplot(1,2,2)
plt.plot(opt_model_fitting.history['loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
#plt.legend(['Training'], loc='upper left')
#plt.savefig('ModelTraining.png')
plt.show()













