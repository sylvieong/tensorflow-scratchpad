import tensorflow as tf
import numpy as np



class TF_DenseLayer(tf.keras.models.Sequential):
    def __init__(self, in_features, out_features):
        super(TF_DenseLayer, self).__init__()

        self.add(tf.keras.layers.Dense(out_features, input_dim=in_features))

        self.add(tf.layers.BatchNormalization())

        # Activation
        self.add(tf.keras.layers.LeakyReLU())

        # Dropout
        self.add(tf.layers.Dropout(rate=0.1))      

class TF_CNN2DLayer(tf.keras.models.Sequential):
    def __init__(self, input_shape, filters, conv_kernel_size, conv_stride=(1, 1), conv_padding='valid', conv_dilation=(1, 1), pool_kernel_size=(2, 2), pool_stride=None, pool_padding=(0, 0), maxpool_enable=False):

        super(TF_CNN2DLayer, self).__init__()
        data_format = 'channels_first'  # optimized for training on NVIDIA GPUs using cuDNN, reference https://www.tensorflow.org/performance/performance_guide#data_formats

        self.add(tf.layers.Conv2D(filters=filters,
                                            kernel_size=conv_kernel_size, strides=conv_stride, padding=conv_padding, data_format=data_format, dilation_rate=conv_dilation, input_shape=input_shape))


        # BatchNorm2d
        self.add(tf.layers.BatchNormalization(axis=1))  # axis=1 bacause con2d channels_first

        # Activation
        self.add(tf.keras.layers.LeakyReLU())

        # MaxPool
        if maxpool_enable:
            self.add(tf.layers.MaxPooling2D(pool_size=pool_kernel_size,
                                                      strides=pool_kernel_size if pool_stride is None else pool_stride,
                                                      padding='same' if pool_padding == (0, 0) else 'valid',
                                                      data_format=data_format))



#------ Recreate VGG-like convnet from https://keras.io/getting-started/sequential-model-guide/ -----
#from keras.layers import Dense, Dropout, Flatten
#from keras.layers import Conv2D, MaxPooling2D
#from keras.optimizers import SGD


model_vgg = tf.keras.models.Sequential()
# input: 100x100 images with 3 channels -> (100, 100, 3) tensors.
# this applies 32 convolution filters of size 3x3 each.

#model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)))
conv0=TF_CNN2DLayer((100, 100, 3), 32, (3,3), pool_kernel_size=(2, 2), maxpool_enable=True)
model_vgg.add(conv0)



#model.add(Conv2D(32, (3, 3), activation='relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))

'''conv1=TF_CNN2DLayer((100,100,32), 32, (3,3), pool_kernel_size=(2, 2), maxpool_enable=True)
model_vgg.add(conv1)
model_vgg.add(tf.keras.layers.Dropout(0.25))
'''

print(model_vgg)

'''
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)

# Generate dummy data
x_train = np.random.random((100, 100, 100, 3)) # dimensions are probably: batch_size, image_rows, image_cols, image_channels
y_train = tf.keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10) #dimensions of size are probably: batchsize, 1
x_test = np.random.random((20, 100, 100, 3))
y_test = tf.keras.utils.to_categorical(np.random.randint(10, size=(20, 1)), num_classes=10)

model.fit(x_train, y_train, batch_size=32, epochs=10)
score = model.evaluate(x_test, y_test, batch_size=32)
'''


#------ Recreate model in basic_graph_modular.py -------
model_final = tf.keras.models.Sequential()

# Dense layer0
#model0 = tf.keras.models.Sequential()
#model0.add(tf.keras.layers.Dense(200, input_shape=(784,), activation='relu'))

model0 = TF_DenseLayer(784, 200)
model_final.add(model0)

# Dense layer1
#model1 = tf.keras.models.Sequential()
#model1.add(tf.keras.layers.Dense(120, input_shape=(200,), activation='relu'))

model1 = TF_DenseLayer(200, 120)
model_final.add(model1)

# Dense layer2
#model2 = tf.keras.models.Sequential()
#model2.add(tf.keras.layers.Dense(10, input_shape=(120,), activation='relu'))

model2 = TF_DenseLayer(120, 10)
model_final.add(model2)


model_final.add(tf.keras.layers.Activation('softmax'))

print(model_final)

 
# train with label, optimization criterion (cross_entropy), optimization method (gradient descent)

model_final.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# Generate dummy data
import numpy as np
data = np.random.random((1000, 784))
labels = np.random.randint(10, size=(1000, 1))

# Convert labels to categorical one-hot encoding
one_hot_labels = tf.keras.utils.to_categorical(labels, num_classes=10)

print(data.shape)
print(labels.shape)
print(one_hot_labels.shape)


# Train the model, iterating on the data in batches of 32 samples
model_final.fit(data, one_hot_labels, epochs=10, batch_size=32)

