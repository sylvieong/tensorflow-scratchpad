import tensorflow as tf
from tensorflow.python.keras.datasets import mnist

# The CSV features in our training & test data
feature_names = [
    'SepalLength',
    'SepalWidth',
    'PetalLength',
    'PetalWidth']

# Let create a dataset for prediction
# We've taken the first 3 examples in FILE_TEST
prediction_input = [[5.9, 3.0, 4.2, 1.5],  # -> 1, Iris Versicolor
                    [6.9, 3.1, 5.4, 2.1],  # -> 2, Iris Virginica
                    [5.1, 3.3, 1.7, 0.5]]  # -> 0, Iris Sentosa

def new_input_fn():
    def decode(x):
        x = tf.split(x, 4)  # Need to split into our 4 features
        return dict(zip(feature_names, x))  # To build a dict of them

    dataset = tf.data.Dataset.from_tensor_slices(prediction_input)
    dataset = dataset.map(decode)
    iterator = dataset.make_one_shot_iterator()
    next_feature_batch = iterator.get_next()
    return next_feature_batch, None  # In prediction, we have no labels


(next_feature, dummy) = new_input_fn()

sess = tf.Session()

for i in range(3):
    feature_value = sess.run(next_feature)
    print('feature_value:')
    print(feature_value)


def mnist_input_fn(components, batch_size=1, perform_shuffle=False, repeat_count=1):
# components should be a tuple (x,y) where x: np_array, y: integer label

    dataset = tf.data.Dataset.from_tensor_slices(components)

    if perform_shuffle:
        # Randomizes input using a window of 256 elements (read into memory)
        dataset = dataset.shuffle(buffer_size=256)
    dataset = dataset.repeat(repeat_count)  # Repeats dataset this # times

    if batch_size > 1:
        dataset = dataset.batch(batch_size)  # Batch size to use

    iterator = dataset.make_one_shot_iterator()
    next_feature, next_label  = iterator.get_next()
    return next_feature, next_label


(x_train, y_train), (x_test, y_test) = mnist.load_data()
print('\n')
print(f'x_train: {x_train.shape}')
print(f'y_train: {y_train.shape}')
print(f'x_test: {x_test.shape}')
print(f'y_test: {y_test.shape}')


# dataset without batching
(next_feature, next_label) = mnist_input_fn((x_train, y_train))

sess = tf.Session()

for i in range(3):
    feature_value, label_value = sess.run((next_feature, next_label))
    print('feature_value.shape:')
    print(feature_value.shape)
    print('label_value.shape:')
    print(label_value.shape)
    print('label_value:')
    print(label_value)


# dataset with batching
(next_feature, next_label) = mnist_input_fn((x_train, y_train),batch_size=32)

sess = tf.Session()

for i in range(3):
    feature_value, label_value = sess.run((next_feature, next_label))
    print('feature_value.shape:')
    print(feature_value.shape)
    print('label_value.shape:')
    print(label_value.shape)
    print('label_value:')
    print(label_value)
