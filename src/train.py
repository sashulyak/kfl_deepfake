import os
import json

import tensorflow as tf

import config


def decode_img(img):
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_png(img, channels=3)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img = tf.image.convert_image_dtype(img, tf.float32)
    # resize the image to the desired size.
    return tf.image.resize(img, [config.IMG_SIZE, config.IMG_SIZE])


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
    # dataset = dataset.shuffle(buffer_size=len(face_paths), seed=SEED)
    # dataset = dataset.repeat()
    dataset = dataset.batch(config.BATCH_SIZE)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset


def resnet_conv_block(input_tensor, kernel_size, filters, stage, block,
                      strides=(2, 2), bias=False):
    filters1, filters2, filters3 = filters

    bn_axis = 3

    conv1_reduce_name = 'conv' + str(stage) + "_" + str(block) + "_1x1_reduce"
    conv1_increase_name = 'conv' + str(stage) + "_" + str(
        block) + "_1x1_increase"
    conv1_proj_name = 'conv' + str(stage) + "_" + str(block) + "_1x1_proj"
    conv3_name = 'conv' + str(stage) + "_" + str(block) + "_3x3"

    x = tf.keras.layers.Conv2D(filters1, (1, 1), strides=strides, use_bias=bias, name=conv1_reduce_name)(input_tensor)
    x = tf.keras.layers.BatchNormalization(axis=bn_axis, name=conv1_reduce_name + "/bn")(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.Conv2D(filters2, kernel_size, padding='same', use_bias=bias, name=conv3_name)(x)
    x = tf.keras.layers.BatchNormalization(axis=bn_axis, name=conv3_name + "/bn")(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.Conv2D(filters3, (1, 1), name=conv1_increase_name, use_bias=bias)(x)
    x = tf.keras.layers.BatchNormalization(axis=bn_axis, name=conv1_increase_name + "/bn")(x)

    shortcut = tf.keras.layers.Conv2D(
        filters3,
        (1, 1),
        strides=strides,
        use_bias=bias,
        name=conv1_proj_name)(input_tensor)
    shortcut = tf.keras.layers.BatchNormalization(axis=bn_axis, name=conv1_proj_name + "/bn")(shortcut)

    x = tf.keras.layers.add([x, shortcut])
    x = tf.keras.layers.Activation('relu')(x)
    return x


def resnet_identity_block(input_tensor, kernel_size, filters, stage, block,
                          bias=False):
    filters1, filters2, filters3 = filters

    bn_axis = 3
    conv1_reduce_name = 'conv' + str(stage) + "_" + str(block) + "_1x1_reduce"
    conv1_increase_name = 'conv' + str(stage) + "_" + str(block) + "_1x1_increase"
    conv3_name = 'conv' + str(stage) + "_" + str(block) + "_3x3"

    x = tf.keras.layers.Conv2D(filters1, (1, 1), use_bias=bias, name=conv1_reduce_name)(
        input_tensor)
    x = tf.keras.layers.BatchNormalization(axis=bn_axis, name=conv1_reduce_name + "/bn")(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.Conv2D(filters2, kernel_size, use_bias=bias, padding='same', name=conv3_name)(x)
    x = tf.keras.layers.BatchNormalization(axis=bn_axis, name=conv3_name + "/bn")(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.Conv2D(filters3, (1, 1), use_bias=bias, name=conv1_increase_name)(x)
    x = tf.keras.layers.BatchNormalization(axis=bn_axis, name=conv1_increase_name + "/bn")(x)

    x = tf.keras.layers.add([x, input_tensor])
    x = tf.keras.layers.Activation('relu')(x)
    return x


def RESNET50(input_shape=None, pooling=None):

    img_input = tf.keras.layers.Input(shape=input_shape)

    bn_axis = 3

    x = tf.keras.layers.Conv2D(
        64, (7, 7), use_bias=False, strides=(2, 2), padding='same',
        name='conv1/7x7_s2')(img_input)
    x = tf.keras.layers.BatchNormalization(axis=bn_axis, name='conv1/7x7_s2/bn')(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = resnet_conv_block(x, 3, [64, 64, 256], stage=2, block=1, strides=(1, 1))
    x = resnet_identity_block(x, 3, [64, 64, 256], stage=2, block=2)
    x = resnet_identity_block(x, 3, [64, 64, 256], stage=2, block=3)

    x = resnet_conv_block(x, 3, [128, 128, 512], stage=3, block=1)
    x = resnet_identity_block(x, 3, [128, 128, 512], stage=3, block=2)
    x = resnet_identity_block(x, 3, [128, 128, 512], stage=3, block=3)
    x = resnet_identity_block(x, 3, [128, 128, 512], stage=3, block=4)

    x = resnet_conv_block(x, 3, [256, 256, 1024], stage=4, block=1)
    x = resnet_identity_block(x, 3, [256, 256, 1024], stage=4, block=2)
    x = resnet_identity_block(x, 3, [256, 256, 1024], stage=4, block=3)
    x = resnet_identity_block(x, 3, [256, 256, 1024], stage=4, block=4)
    x = resnet_identity_block(x, 3, [256, 256, 1024], stage=4, block=5)
    x = resnet_identity_block(x, 3, [256, 256, 1024], stage=4, block=6)

    x = resnet_conv_block(x, 3, [512, 512, 2048], stage=5, block=1)
    x = resnet_identity_block(x, 3, [512, 512, 2048], stage=5, block=2)
    x = resnet_identity_block(x, 3, [512, 512, 2048], stage=5, block=3)

    x = tf.keras.layers.AveragePooling2D((7, 7), name='avg_pool')(x)

    if pooling == 'avg':
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
    elif pooling == 'max':
        x = tf.keras.layers.GlobalMaxPooling2D()(x)

    inputs = img_input

    model = tf.keras.models.Model(inputs, x, name='vggface_resnet50')

    model.load_weights(config.BACKBONE_WEIGHTS_PATH)

    return model


if __name__ == '__main__':
    vggface = RESNET50(input_shape=(config.IMG_SIZE, config.IMG_SIZE, 3), pooling='max')
    vggface_out = vggface.get_layer('avg_pool').output
    flatten = tf.keras.layers.Flatten(name='flatten')(vggface_out)
    out = tf.keras.layers.Dense(1, activation='sigmoid')(flatten)
    model = tf.keras.models.Model(vggface.input, out)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

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
        epochs=config.POCHS
    )
