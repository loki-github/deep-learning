import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
tfds.disable_progress_bar()

print("Using:")
print('\t\u2022 Tensor flow version:', tf.__version__)
print('\t\u2022 tf.keras version:', tf.keras.__version__)
print('\t\u2022 Running on GPU' if tf.test.is_gpu_available() else '\t\u2022 GPU device not found. Running on CPU')

training_set, dataset_info = tfds.load('mnist', split = 'train', as_supervised = True, with_info = True)

num_classes = dataset_info.features['label'].num_classes
print('There are {} classes in our dataset'.format(num_classes))

num_training_examples = dataset_info.splits['train'].num_examples
print('There are {:,} images in the training set'.format(num_training_examples))

for image, label in training_set.take(1):
    print('The images in the training set have:')
    print('dtype:', image.dtype)
    print('shape:', image.shape)

    print('The label of the images have:')
    print('dtype:', label.dtype)

for image, label in training_set.take(1):
    image = image.numpy().squeeze()
    label = label.numpy()

print('The label of this image is:',label)

def normalize(image, label):
    image = tf.cast(image, tf.float32)
    image = image / 255
    return image, label

batch_size = 64

training_batches = training_set.cache().shuffle(num_training_examples//4).batch(batch_size).map(normalize).prefetch(1)

for image_batch, label_batch in training_batches.take(1):
    print('The images in each batch have:')
    print('\u2022 dtype:', image_batch.dtype)
    print('\u2022 shape:', image_batch.shape)

    print('\nThere are a total of {} image labels in this batch:'.format(label_batch.numpy().size))
    print(label_batch.numpy())

for image_batch, label_batch in training_batches.take(1):
    images = image_batch.numpy().squeeze()
    print(images.shape)
    labels = label_batch.numpy()

def sigmoid(x):
    return 1/ (1+tf.exp(x))


inputs = tf.reshape(images, [images.shape[0],-1])
print(inputs.shape)
tf.random.set_seed(7)
n_input = inputs.shape[1]
n_hidden = 256
n_output = 10
weights_input_hidden = tf.random.normal((n_input, n_hidden))
weights_hidden_output = tf.random.normal((n_hidden,n_output))
B_hidden = tf.random.normal((1, n_hidden))
B_output = tf.random.normal((1, n_output))

hidden_output = sigmoid(tf.matmul(inputs,weights_input_hidden) + B_hidden)

output = tf.matmul(hidden_output, weights_hidden_output) + B_output

exp_sum_each_col = tf.reduce_sum(tf.exp(output),axis=1,keepdims=True)
probabilities = tf.divide(tf.exp(output), exp_sum_each_col)

print(exp_sum_each_col)



