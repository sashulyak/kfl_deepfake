import os
import json

import tensorflow as tf

import config


def decode_img(img):
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, [config.IMG_SIZE, config.IMG_SIZE])
    return tf.keras.applications.inception_resnet_v2.preprocess_input(img)


def read_image(file_path, label):
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, label


def get_dataset(metadata_path, train_faces_dir):
    with open(metadata_path) as json_file:
        metadata = json.load(json_file)

    all_face_file_names = os.listdir(train_faces_dir)
    metadata_face_file_names = list(metadata.keys())
    face_file_names = list(set(all_face_file_names) & set(metadata_face_file_names))

    labels = []
    face_paths = []
    for face_file_name in face_file_names:
        labels.append(int(metadata[face_file_name]['label']))
        face_paths.append(os.path.join(train_faces_dir, face_file_name))

    paths_tensor = tf.constant(face_paths)
    labels_tensor = tf.constant(labels)
    dataset = tf.data.Dataset.from_tensor_slices((paths_tensor, labels_tensor))
    dataset = dataset.map(read_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(config.BATCH_SIZE)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset


if __name__ == '__main__':

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        inception_resnet_v2 = tf.keras.applications.inception_resnet_v2.InceptionResNetV2(
            include_top=False, weights='imagenet',
            input_shape=(config.IMG_SIZE, config.IMG_SIZE, 3),
            pooling='max')
        inception_resnet_v2_out = inception_resnet_v2.get_layer('global_max_pooling2d').output

        out = tf.keras.layers.Dense(1, activation='sigmoid')(inception_resnet_v2_out)
        model = tf.keras.models.Model(inception_resnet_v2.input, out)
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
