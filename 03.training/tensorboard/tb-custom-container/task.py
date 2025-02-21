
import logging
import os

import tensorflow as tf
import tensorflow_datasets as tfds

IMG_WIDTH = 128

def normalize_img(image):
    """Normalizes image.

    * Resizes image to IMG_WIDTH x IMG_WIDTH pixels
    * Casts values from `uint8` to `float32`
    * Scales values from [0, 255] to [0, 1]

    Returns:
      A tensor with shape (IMG_WIDTH, IMG_WIDTH, 3). (3 color channels)
    """
    image = tf.image.resize_with_pad(image, IMG_WIDTH, IMG_WIDTH)
    return image / 255.


def normalize_img_and_label(image, label):
    """Normalizes image and label.

    * Performs normalize_img on image
    * Passes through label unchanged

    Returns:
      Tuple (image, label) where
      * image is a tensor with shape (IMG_WIDTH, IMG_WIDTH, 3). (3 color
        channels)
      * label is an unchanged integer [0, 4] representing flower type
    """
    return normalize_img(image), label

logging.info('Loading and preprocessing data ...')
dataset = tfds.load('tf_flowers:3.*.*',
                    split='train',
                    try_gcs=True,
                    shuffle_files=True,
                    as_supervised=True)
dataset = dataset.map(normalize_img_and_label,
                      num_parallel_calls=tf.data.experimental.AUTOTUNE)
dataset = dataset.cache()
dataset = dataset.shuffle(1000)
dataset = dataset.batch(128)
dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

logging.info('Creating and training model ...')

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16,
                           3,
                           padding='same',
                           activation='relu',
                           input_shape=(IMG_WIDTH, IMG_WIDTH, 3)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation="relu"),
    tf.keras.layers.Dense(5)  # 5 classes
])

logging.info('Compiling model ...')
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])

log_dir = "logs"
if 'AIP_TENSORBOARD_LOG_DIR' in os.environ:
  log_dir=os.environ['AIP_TENSORBOARD_LOG_DIR']

tensorboard_callback = tf.keras.callbacks.TensorBoard(
  log_dir=log_dir,
  histogram_freq=1)

logging.info('Training model ...')
model.fit(dataset, epochs=13, callbacks=[tensorboard_callback])

logging.info('Model training done')
