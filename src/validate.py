import os
import json
from typing import List

import numpy as np
import tensorflow as tf
from sklearn.metrics import log_loss

import config
from utils import read_faces_from_video


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
        self.broken_files = []

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

            try:
                video_frames = read_faces_from_video(video_path, img_size=config.IMG_SIZE)
            except Exception:
                self.broken_files.append(filename)
                video_frames = np.zeros(shape=(self.frames_per_movie, config.IMG_SIZE, config.IMG_SIZE, 3), dtype=np.uint8)

            if len(video_frames) < self.frames_per_movie:
                for i in range(self.frames_per_movie - len(video_frames)):
                    video_frames.append(video_frames[-1])
            x.extend(video_frames)
        x = np.array(x) / 255.0

        return x


if __name__ == '__main__':
    video_file_names, labels, groups = get_video_names(config.FACES_VAL_METADATA_PATH, config.TRAIN_FACES_DIR)

    data_generator = FacesDataGenerator(
        video_file_names=video_file_names,
        video_groups=groups,
        videos_directory=config.TRAIN_VIDEOS_DIR,
        batch_size=config.BATCH_SIZE,
        frames_per_movie=config.FRAMES_PER_VIDEO,
        image_size=config.IMG_SIZE)

    model = tf.keras.models.load_model(config.MODEL_PATH)
    model.run_eagerly = False

    predictions = model.predict(
        data_generator,
        verbose=1,
        workers=16,
        use_multiprocessing=True,
        max_queue_size=30)

    predictions_grouped = np.reshape(predictions, (len(video_file_names), config.FRAMES_PER_VIDEO))
    predictions_mean = np.mean(predictions_grouped, axis=1)

    broken_file_indexes = np.isin(video_file_names, data_generator.broken_files)
    predictions_mean[broken_file_indexes] = 0.5

    val_loss = log_loss(labels.astype(float), predictions_mean.astype(float))
    print('Validation loss:', val_loss)

    # Validation loss: 0.5442721299660042