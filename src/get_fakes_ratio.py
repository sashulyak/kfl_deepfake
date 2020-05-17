import os
import json

import numpy as np
from sklearn.metrics import log_loss

import config


def get_labels(metadata_path, train_faces_dir):
    with open(metadata_path) as json_file:
        metadata = json.load(json_file)

    all_face_file_names = os.listdir(train_faces_dir)
    metadata_face_file_names = list(metadata.keys())
    face_file_names = list(set(all_face_file_names) & set(metadata_face_file_names))

    labels = []
    for face_file_name in face_file_names:
        labels.append(metadata[face_file_name]['label'])

    return np.array(labels)


train_labels = get_labels(config.FACES_TRAIN_METADATA_PATH, config.TRAIN_FACES_DIR)
val_labels = get_labels(config.FACES_VAL_METADATA_PATH, config.TRAIN_FACES_DIR)

all_labels = np.concatenate((train_labels, val_labels))

train_fake_ratio = np.sum(train_labels) / train_labels.shape[0]
val_fake_ratio = np.sum(val_labels) / val_labels.shape[0]
all_fake_ratio = np.sum(all_labels) / all_labels.shape[0]

train_fake_ratio_loss = log_loss(val_labels, np.ones_like(val_labels) * train_fake_ratio)
val_fake_ratio_loss = log_loss(val_labels, np.ones_like(val_labels) * val_fake_ratio)
all_fake_ratio_loss = log_loss(val_labels, np.ones_like(val_labels) * all_fake_ratio)


print('Train fake ratio: {} Train fake ratio loss: {}'.format(train_fake_ratio, train_fake_ratio_loss))
print('Val fake ratio: {} Val fake ratio loss: {}'.format(val_fake_ratio, val_fake_ratio_loss))
print('All fake ratio: {} All fake ratio loss: {}'.format(all_fake_ratio, all_fake_ratio_loss))

'''
Train fake ratio: 0.8293777073724644 Train fake ratio loss: 0.4668777079910673
Val fake ratio: 0.8230496060370658 Val fake ratio loss: 0.46673757983381214
All fake ratio: 0.8281013933187846 All fake ratio loss: 0.4668265350830934
'''


def get_train_data_lists():
    chunks_list = os.listdir(config.TRAIN_VIDEOS_DIR)
    labels = []

    for chunk_dir_name in chunks_list:
        chunk_dir = os.path.join(config.TRAIN_VIDEOS_DIR, chunk_dir_name)
        chunk_file_names = os.listdir(chunk_dir)
        chunk_file_names = [name for name in chunk_file_names if name != 'metadata.json']

        metadata_file_path = os.path.join(chunk_dir, 'metadata.json')
        with open(metadata_file_path) as json_file:
            chunk_metadata = json.load(json_file)

        for video_file_name in chunk_file_names:
            labels.append(chunk_metadata[video_file_name]['label'] == 'FAKE')

    return np.array(labels)


labels = get_train_data_lists()
fake_ratio = np.sum(labels) / labels.shape[0]
fake_ratio_loss = log_loss(val_labels, np.ones_like(val_labels) * fake_ratio)
print('Fake ratio: {} Train fake ratio loss: {}'.format(fake_ratio, fake_ratio_loss))

'''Fake ratio: 0.839239252681584 Train fake ratio loss: 0.4676838854574761'''