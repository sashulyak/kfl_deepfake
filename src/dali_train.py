import os
import json
from typing import Tuple

import numpy as np
import tensorflow as tf

import config as config
from dali_dataset import get_dali_dataset


def get_train_data_lists(train_videos_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Collect video paths and labels across all training groups.

    :param train_videos_dir: path to train data directory
    :return: list of train video paths, video file names, corresponding labels and groups (names of folders)
    """
    chunks_list = os.listdir(train_videos_dir)

    paths = []
    labels = []
    groups = []
    names = []
    for chunk_dir_name in chunks_list:
        chunk_dir = os.path.join(train_videos_dir, chunk_dir_name)
        chunk_file_names = os.listdir(chunk_dir)
        chunk_file_names = [name for name in chunk_file_names if name != 'metadata.json']

        metadata_file_path = os.path.join(chunk_dir, 'metadata.json')
        with open(metadata_file_path) as json_file:
            chunk_metadata = json.load(json_file)

        for video_file_name in chunk_file_names:
            paths.append(os.path.join(chunk_dir, video_file_name))
            labels.append(chunk_metadata[video_file_name]['label'] == 'FAKE')
            groups.append(chunk_dir_name)
            names.append(video_file_name)

    return np.array(paths), np.array(names), np.array(labels), np.array(groups)


def get_dataset() -> tf.data.Dataset:
    """
    Create Tensorflow dataset consisted of face crop/label pairs.

    :param metadata_path: path to metadata file which stores information about cropped faces
    :param train_faces_dir: path to directory where face crops are stored
    :return: Tensorflow dataset
    """
    video_paths, video_file_names, labels, groups = get_train_data_lists(config.TRAIN_VIDEOS_DIR)

    dataset = get_dali_dataset(video_paths, labels)
    dataset = dataset.shuffle(seed=42)
    dataset = dataset.batch(config.BATCH_SIZE)


if __name__ == '__main__':

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        xception = tf.keras.applications.xception.Xception(
            include_top=False, weights='imagenet',
            input_shape=(config.IMG_SIZE, config.IMG_SIZE, 3),
            pooling='max')
        xception_out = xception.get_layer('global_max_pooling2d').output

        out = tf.keras.layers.Dense(1, activation='sigmoid')(xception_out)
        model = tf.keras.models.Model(xception.input, out)
        model.compile(optimizer='adam', loss='binary_crossentropy')

    train_dataset = get_dataset()

    model.fit(
        x=train_dataset,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(config.MODEL_PATH, verbose=1, save_best_only=True),
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, mode='auto', restore_best_weights=True)
        ],
        epochs=config.EPOCHS
    )
