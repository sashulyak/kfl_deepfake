import os

import tensorflow as tf

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
        epochs=config.EPOCHS
    )
