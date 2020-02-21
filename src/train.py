import os
import json

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, LSTM, TimeDistributed
from tqdm import tqdm

import config


def decode_img(img):
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, [config.IMG_SIZE, config.IMG_SIZE])
    return tf.keras.applications.xception.preprocess_input(img)


def read_frames(file_paths, label):
    frames = tf.map_fn(tf.io.read_file, file_paths)
    frames_decoded = tf.map_fn(decode_img, frames, dtype=tf.dtypes.float32)
    return frames_decoded, label


def get_dataset(metadata_path, train_faces_dir):
    with open(metadata_path) as json_file:
        metadata = json.load(json_file)

    all_face_file_names = os.listdir(train_faces_dir)
    metadata_face_file_names = list(metadata.keys())
    face_file_names = list(set(all_face_file_names) & set(metadata_face_file_names))

    labels = []
    face_paths = []
    video_names = []
    for face_file_name in face_file_names:
        labels.append(int(metadata[face_file_name]['label']))
        face_paths.append(os.path.join(train_faces_dir, face_file_name))
        video_names.append(metadata[face_file_name]['video'])

    labels = np.array(labels)
    face_paths = np.array(face_paths)
    video_names = np.array(video_names)

    unique_video_names = np.unique(video_names)

    print('Prepare dataset ...')
    face_paths_grouped = []
    labels_grouped = []
    for video_name in tqdm(unique_video_names):
        video_index = video_names == video_name
        video_faces = face_paths[video_index]
        if len(video_faces) == config.FRAMES_PER_VIDEO:
            face_paths_grouped.append(sorted(video_faces))
            labels_grouped.append(labels[video_index][0])

    paths_tensor = tf.constant(face_paths_grouped)
    labels_tensor = tf.constant(labels_grouped)
    dataset = tf.data.Dataset.from_tensor_slices((paths_tensor, labels_tensor))
    dataset = dataset.map(read_frames, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(config.BATCH_SIZE)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset


if __name__ == '__main__':

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
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

    train_dataset = get_dataset(config.FACES_TRAIN_METADATA_PATH, config.TRAIN_FACES_DIR)
    val_dataset = get_dataset(config.FACES_VAL_METADATA_PATH, config.TRAIN_FACES_DIR)

    model.fit(
        x=train_dataset,
        validation_data=val_dataset,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(config.MODEL_PATH, verbose=1, save_best_only=True),
            tf.keras.callbacks.CSVLogger(config.LOG_PATH, append=True, separator=';'),
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, mode='auto', restore_best_weights=True)
        ],
        class_weight={
            0: config.TRAIN_FAKE_RATIO,
            1: 1. - config.TRAIN_FAKE_RATIO
        },
        epochs=config.EPOCHS
    )
