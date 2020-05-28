import os

import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, LSTM, TimeDistributed

import config as config
from dali_dataset import get_dali_dataset


def get_dataset() -> tf.data.Dataset:
    """
    Create test Tensorflow dataset.

    :return: Tensorflow dataset
    """
    video_names = [
        '01.mp4',
        '02.mp4',
        '03.mp4',
        '04.mp4',
        '05.mp4',
        '06.mp4',
        '07.mp4',
        '08.mp4',
        '09.mp4',
        '10.mp4']
    labels = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    video_paths = []
    for video_name in video_names:
        video_paths.append(os.path.join(config.BASE_DIR, 'dali_test_sample', video_name))

    dataset = get_dali_dataset(video_paths, labels)
    dataset = dataset.batch(1)
    return dataset


if __name__ == '__main__':
    video_input = Input((config.FRAMES_PER_VIDEO, config.IMG_SIZE, config.IMG_SIZE, 3))
    xception = tf.keras.applications.xception.Xception(
        include_top=False, weights='imagenet',
        input_shape=(config.IMG_SIZE, config.IMG_SIZE, 3),
        pooling='max')
    time_distributed = TimeDistributed(xception)(video_input)
    lstm = LSTM(128, dropout=0.2, recurrent_dropout=0.2)(time_distributed)
    out = Dense(1, activation='sigmoid')(lstm)
    model = tf.keras.models.Model(video_input, out)
    model.compile(optimizer='adam', loss='binary_crossentropy')

    train_dataset = get_dataset()

    model.fit(
        x=train_dataset,
        epochs=config.EPOCHS
    )
