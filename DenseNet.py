from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import AveragePooling2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers.core import Dense, Dropout, Activation
from keras.regularizers import l2
from keras.models import Model
from keras.layers import Input, Concatenate
import keras.backend as K


def denseblock_unit(X, add_axis, num_filter, dropout_rate=0.2, weight_decay=1e-4):
	'''
	This function is a unit architecture of denseblock.
	BN -> ReLU -> Conv2D -> Dropout -> BN -> ReLU -> Conv2D -> Dropout

	Arguments:
	X -- Keras network 
	add_axis -- (int) contatenate axis
	num_filter -- (int) number of filters
	dropout_rate -- (float) 
	weight_decay -- (float)

	Returns:
	X -- Keras network  
	'''

	#1x1 Conv
	X = BatchNormalization(axis=add_axis, gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay))(X)
	X = Activation('relu')(X)
	X = Conv2D(num_filter, (1, 1), kernel_initializer='glorot_uniform', padding='same', use_bias=False, kernel_regularizer=l2(weight_decay))(X)
	if dropout_rate:
		X = Dropout(dropout_rate)(X)

	#3x3 Conv
	X = BatchNormalization(axis=add_axis, gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay))(X)
	X = Activation('relu')(X)
	X = Conv2D(num_filter, (3, 3), kernel_initializer='glorot_uniform', padding='same', use_bias=False, kernel_regularizer=l2(weight_decay))(X)
	if dropout_rate:
		X = Dropout(dropout_rate)(X)

	return X


def transition_layer(X, add_axis, num_filter, dropout_rate=0.2, weight_decay=1e-4):
	'''
	This function aim to connect two denseblock.
	BN -> ReLU -> Conv2D -> Dropout -> Ave Pooling

	Arguments:
	X -- Keras network 
	add_axis -- (int) contatenate axis
	num_filter -- (int) number of filters
	dropout_rate -- (float) 
	weight_decay -- (float)

	Returns:
	X -- Keras network  
	'''

	X = BatchNormalization(axis=add_axis, gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay))(X)
	X = Activation('relu')(X)
	X = Conv2D(num_filter, (1, 1), kernel_initializer='glorot_uniform', padding='same', use_bias=False, kernel_regularizer=l2(weight_decay))(X)
	if dropout_rate:
		X = Dropout(dropout_rate)(X)
	X = AveragePooling2D((2, 2), strides=(2, 2), padding='same')(X)

	return X


def denseblock(X, add_axis, num_layers, num_filter, growth_rate, dropout_rate=0.2, weight_decay=1e-4):
	'''
	This function aim to build a complete denseblock by connect every denseblock unit.

	Arguments:
	X -- Keras network
	add_axis -- (int) contatenate axis
	num_layers -- (int) number of denseblock unit in this denseblock, based on growth rate
	num_filter -- (int) number of filter
	growth_rate -- (int) 'k' in 'Densely Connected Convolutional Networks', where num_filter = k0 + k*(l-1)
	dropout_rate -- (float) 
	weight_decay -- (float)

	Returns:
	X -- Keras network  
	'''

	#temporary list that will concatenate in every loop
	connect_list = [X] 

	for i in range(num_layers):
		X = denseblock_unit(X, add_axis, num_layers, dropout_rate, weight_decay)
		connect_list.append(X)
		X = Concatenate(axis=add_axis)(connect_list)
		#Calculate number of filter in the next denseblock
		num_filter += growth_rate

	return X, num_filter


def densenet(img_shape, num_classes, num_filter, growth_rate, eve_layers=(6, 12, 24, 16), num_denseblock=4, dropout_rate=0.2, weight_decay=1e-4):
	'''
	This function aim to build a complete DenseNet.
	DenseNet-121 -->> eve_layers=(6, 12, 24, 16)
	DenseNet-169 -->> eve_layers=(6, 12, 32, 32)
	DenseNet-201 -->> eve_layers=(6, 12, 48, 32)
	DenseNet-264 -->> eve_layers=(6, 12, 64, 48)

	Arguments:
	img_shape -- (tuple or list) (height, width, channel)
	num_classes -- number of classes
	num_filter -- (int) initial number of filter (equal 2*growth_rate)
	growth_rate -- (int) 'k' in 'Densely Connected Convolutional Networks', where num_filter = k0 + k*(l-1)
	eve_layers -- number of unit in four denseblock
	num_denseblock -- number of denseblock (this four model all have 4 block)
	dropout_rate -- (float) 
	weight_decay -- (float)

	Returns:
	model -- Keras model 
	'''

	#if Keras backend is 'theano', add_axis is 1, if backend is 'tensorflow', add_axis is -1
	if K.image_dim_ordering() == 'th':
		add_axis = 1
	elif K.image_dim_ordering() == 'tf':
		add_axis = -1

	X_input = Input(shape=img_shape)

	#First Convolution
	X = Conv2D(num_filter, (7, 7), strides=(2, 2), kernel_initializer='glorot_uniform', padding='same', name='first_conv2D', use_bias=False, kernel_regularizer=l2(weight_decay))(X_input)
	X = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', data_format=None)(X)
	#Add first three denseblocks
	for i in range(num_denseblock-1):
		X, num_filter = denseblock(X, add_axis, eve_layers[i], num_filter, growth_rate, dropout_rate=dropout_rate, weight_decay=weight_decay)

		#Add transition layer
		X = transition_layer(X, add_axis, num_filter, dropout_rate=dropout_rate, weight_decay=weight_decay)

	#Add last denseblocks without transition layer
	X, num_filter = denseblock(X, add_axis, eve_layers[-1], num_filter, growth_rate, dropout_rate=dropout_rate, weight_decay=weight_decay)

	#Add Classification Layer
	X = BatchNormalization(axis=add_axis, gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay))(X)
	X = Activation('relu')(X)
	X = GlobalAveragePooling2D(data_format=K.image_data_format())(X)
	X = Dense(num_classes, activation='softmax', kernel_regularizer=l2(weight_decay), bias_regularizer=l2(weight_decay))(X)

	model = Model(inputs=X_input, outputs=X, name='DenseNet')

	return model