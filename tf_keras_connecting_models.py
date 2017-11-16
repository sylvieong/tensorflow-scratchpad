import tensorflow as tf
import numpy as np



class TF_DenseLayer(tf.keras.models.Sequential):
    def __init__(self, input_dim, out_dim):
        super(TF_DenseLayer, self).__init__()
        self.add(tf.keras.layers.Dense(out_dim, input_dim=input_dim, activation='relu'))
      

# Recreate model in basic_graph_modular.py
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

# TODO: 
# how to train with label, optimization criterion (cross_entropy), optimization method (gradient descent)

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



'''

# Example
# as first layer in a sequential model:
model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Dense(32, input_shape=(16,)))

#model.add(Dense(64, activation='tanh'))

# now the model will take as input arrays of shape (*, 16)
# and output arrays of shape (*, 32)

# after the first layer, you don't need to specify
# the size of the input anymore:
model.add(tf.keras.layers.Dense(32))

print(model)

model_final = tf.keras.models.Sequential()

model_final.add(model)


print(model_final)
'''
