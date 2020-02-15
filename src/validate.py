import os
import json
from typing import List

import numpy as np
import tensorflow as tf
from sklearn.metrics import log_loss

import config
from utils import read_faces_from_video, get_corrupted_file_indexes


def get_video_names(metadata_path, train_faces_dir):
    with open(metadata_path) as json_file:
        metadata = json.load(json_file)

    all_face_file_names = os.listdir(train_faces_dir)
    metadata_face_file_names = list(metadata.keys())
    face_file_names = list(set(all_face_file_names) & set(metadata_face_file_names))

    labels = []
    video_paths = []
    groups = []
    for face_file_name in face_file_names:
        labels.append(int(metadata[face_file_name]['label']))
        video_paths.append(metadata[face_file_name]['video'])
        groups.append(metadata[face_file_name]['group'])

    return np.array(video_paths), np.array(labels), np.array(groups)


class FacesDataGenerator(tf.keras.utils.Sequence):
    def __init__(
            self,
            video_file_names: List[str],
            videos_directory: str,
            video_groups: List[str] = None,
            batch_size: int = 64,
            frames_per_movie: int = 3,
            image_size: int = 200):
        self.batch_size = batch_size
        self.frames_per_movie = frames_per_movie
        self.image_size = image_size
        self.video_file_names = video_file_names
        self.videos_directory = videos_directory
        self.video_groups = video_groups

    def __len__(self) -> int:
        return int(np.ceil(len(self.video_file_names) / self.batch_size))

    def __getitem__(self, index) -> tuple:
        x = []
        batch_start = index * self.batch_size
        batch_end = min([(index + 1) * self.batch_size, len(video_file_names)])
        for i in range(batch_start, batch_end):
            filename = self.video_file_names[i]
            if self.video_groups is not None:
                group_path = os.path.join(self.videos_directory, self.video_groups[i])
                video_path = os.path.join(group_path, filename)
            else:
                video_path = os.path.join(self.videos_directory, filename)

            video_frames = read_faces_from_video(video_path, img_size=config.IMG_SIZE)

            if len(video_frames) < self.frames_per_movie:
                for i in range(self.frames_per_movie - len(video_frames)):
                    video_frames.append(video_frames[-1])
            x.extend(video_frames)
        x = np.array(x) / 255.0

        return x


if __name__ == '__main__':
    video_file_names, labels, groups = get_video_names(config.FACES_VAL_METADATA_PATH, config.TRAIN_FACES_DIR)

    corrupted_file_indexes = get_corrupted_file_indexes(
        video_file_names,
        config.TRAIN_VIDEOS_DIR,
        video_groups=groups,
        verbose=True)

    good_file_names = video_file_names[np.invert(corrupted_file_indexes)]
    good_groups = groups[np.invert(corrupted_file_indexes)]

    data_generator = FacesDataGenerator(
        video_file_names=good_file_names,
        video_groups=good_groups,
        videos_directory=config.TRAIN_VIDEOS_DIR,
        batch_size=config.BATCH_SIZE,
        frames_per_movie=config.FRAMES_PER_VIDEO,
        image_size=config.IMG_SIZE)

    model = tf.keras.models.load_model(config.MODEL_PATH)
    predictions = model.predict(
        data_generator,
        verbose=1,
        workers=16,
        use_multiprocessing=True,
        max_queue_size=30)

    predictions_grouped = np.reshape(predictions, (len(good_file_names), config.FRAMES_PER_VIDEO))
    predictions_mean = np.mean(predictions_grouped, axis=1)

    all_predictions = np.empty(shape=(len(video_file_names)), dtype=float)
    all_predictions[np.invert(corrupted_file_indexes)] = predictions_mean
    all_predictions[corrupted_file_indexes] = 0.5
    all_predictions = np.clip(all_predictions, 0.1, 0.9)

    val_loss = log_loss(labels, all_predictions)
    print('Validation loss:', val_loss)

    # Validation loss: 0.5644014298127646
