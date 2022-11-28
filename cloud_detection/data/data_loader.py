import tensorflow as tf


class DataLoader(object):
    """
    Data pipeline from data_reader (image, label) to tf.data.
    """

    def __init__(self, data_reader, image_size=384):
        self.data_reader = data_reader
        self.image_size = image_size

    def __call__(self, batch_size=8, train_dataset=True):

        if train_dataset:
            dataset = tf.data.Dataset.from_generator(
                self.data_reader.iteration,
                output_types=(tf.float32, tf.float32),
                output_shapes=([self.image_size, self.image_size, 4],
                               [self.image_size, self.image_size, 1])
            )
            dataset = dataset.repeat().batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
            return dataset
        else:
            dataset = tf.data.Dataset.from_generator(
                self.data_reader.iteration,
                output_types=(tf.float32),
                output_shapes=([self.image_size, self.image_size, 4])
            )
            dataset = dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
            return dataset
