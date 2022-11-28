import sys
import os
import time

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split

import configs.config
import data.data_loader
import data.data_reader
import utils.image_utils
import utils.tracker
import modules.cloud_net
import modules.optimizer
import modules.losses


def get_dataset():
    train_data_reader = data.data_reader.DataReader(
        train_image_split,
        train_mask_split,
        image_size=configs.config.params['image_size'],
        augment=True
    )
    train_dataset = data.data_loader.DataLoader(
        train_data_reader,
        configs.config.params['image_size'])(batch_size=configs.config.params['batch_size'])
    train_dataset.len = len(train_data_reader)

    valid_data_reader = data.data_reader.DataReader(
        validation_image_split,
        validation_mask_split,
        image_size=configs.config.params['image_size'],
        augment=False
    )
    valid_dataset = data.data_loader.DataLoader(
        valid_data_reader,
        configs.config.params['image_size']
    )(batch_size=configs.config.params['batch_size'])
    valid_dataset.len = len(valid_data_reader)

    return train_dataset, valid_dataset


def get_model(input_row, input_col, model_name='cloud'):
    if model_name == 'cloud':
        return modules.cloud_net.model_arch(input_row, input_col)
    else:
        return ValueError('Unsupported model {}'.format(model_name))


def train():
    train_dataset, valid_dataset = get_dataset()
    train_steps_per_epoch = np.ceil(train_dataset.len / configs.config.params['batch_size'])
    valid_step_per_epoch = np.ceil(valid_dataset.len / configs.config.params['batch_size'])
    cosine_steps = configs.config.params['cosine_epochs'] * train_steps_per_epoch

    input_row = input_col = configs.config.params['image_size']
    model = get_model(input_row, input_col)

    schedule = tf.keras.experimental.CosineDecay(
        configs.config.params['init_learning_rate'],
        cosine_steps,
        alpha=configs.config.params['cosine_alpha']
    )
    optim = modules.optimizer.Optimizer('adam', schedule=schedule)()
    model.compile(optimizer=optim, loss=modules.losses.jaccard_loss, metrics=[modules.losses.jaccard_loss])
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(configs.config.params['saved_model_dir'], 'Cloud.{epoch:03d}-{val_loss:.6f}.h5'),
        monitor='val_loss',
        save_best_only=True
    )

    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=9, verbose=1)
    csv_logger = tf.keras.callbacks.CSVLogger(
        'training_log_{}.log'.format(time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())))
    model.fit(
        train_dataset,
        validation_data=valid_dataset,
        epochs=configs.config.params['n_epochs'],
        steps_per_epoch=train_steps_per_epoch,
        validation_steps=valid_step_per_epoch,
        verbose=1,
        callbacks=[
            early_stop,
            csv_logger,
            model_checkpoint,
            utils.tracker.ADAMLearningRateTracker()
        ]
    )


if __name__ == '__main__':
    np.random.seed(configs.config.params['random_seed'])
    tf.random.set_seed(configs.config.params['random_seed'])

    GLOBAL_PATH = configs.config.params['train_dataset_dir']
    TRAIN_FOLDER = os.path.join(GLOBAL_PATH, '38-Cloud_training')
    TEST_FOLDER = os.path.join(GLOBAL_PATH, '38-Cloud_test')
    train_patches_csv_name = 'training_patches_38-cloud_nonempty.csv'
    df_train_image = pd.read_csv(os.path.join(GLOBAL_PATH, train_patches_csv_name))

    validation_ratio = configs.config.params['validation_ratio']
    train_image, train_mask = utils.image_utils.get_input_image_names(df_train_image, TRAIN_FOLDER, if_train=True)
    train_image_split, validation_image_split, train_mask_split, validation_mask_split = train_test_split(
        train_image,
        train_mask,
        test_size=validation_ratio,
        random_state=configs.config.params['random_seed'],
        shuffle=True
    )

    train()
