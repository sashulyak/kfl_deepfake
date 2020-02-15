import os
import json
from multiprocessing import Pool
from typing import List, Tuple

import cv2
import cvlib as cv
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from tqdm import tqdm

import config
from utils import extend_rect_to_square, read_faces_from_video


def get_train_data_lists() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    chunks_list = os.listdir(config.TRAIN_VIDEOS_DIR)

    paths = []
    labels = []
    groups = []
    names = []
    for chunk_dir_name in chunks_list:
        chunk_dir = os.path.join(config.TRAIN_VIDEOS_DIR, chunk_dir_name)
        chunk_file_names = os.listdir(chunk_dir)
        chunk_file_names = [name for name in chunk_file_names if name != 'metadata.json']

        metadata_file_path = os.path.join(chunk_dir, 'metadata.json')
        with open(metadata_file_path) as json_file:
            chunk_metadata = json.load(json_file

        for video_file_name in chunk_file_names:
            paths.append(os.path.join(chunk_dir, video_file_name))
            labels.append(chunk_metadata[video_file_name]['label'] == 'FAKE')
            groups.append(chunk_dir_name)
            names.append(video_file_name)

    return np.array(paths), np.array(names), np.array(labels), np.array(groups)


def get_image_name_from_video_name(video_name, frame_number=None):
    if frame_number is not None:
        return video_name[:-4] + '_{}.png'.format(frame_number)
    else:
        return video_name[:-4] + '.png'


def save_faces_from_video(path: str) -> List[str]:
    video_name = os.path.basename(path)
    image_paths = []
    for i in range(FRAMES_PER_VIDEO):
        image_name = get_image_name_from_video_name(video_name, i)
        image_paths.append(os.path.join(config.TRAIN_FACES_DIR, image_name))

    for image_path in image_paths:
        if os.path.exists(image_path):
            return image_paths

    try:
        faces_to_save = read_faces_from_video(path, swap_channels=False)
    except Exception:
            print('Cannot read video:', path)

    for image_path, face in zip(image_paths, faces_to_save):
        try:
            cv2.imwrite(image_path, face)
        except Exception:
            print('Cannot save face image:', path, face.shape)
    return image_paths


def extract_faces_from_videos_parallel(paths: List[str]) -> np.array:
    os.makedirs(TRAIN_FACES_DIR, exist_ok=True)

    faces = []
    with Pool() as p:
        with tqdm(total=len(paths)) as pbar:
            for i, video_faces_paths in enumerate(p.imap_unordered(save_faces_from_video, paths)):
                faces.extend(video_faces_paths)
                pbar.update()
    return np.array(faces)


def save_faces_metadata_to_json(video_paths, labels, groups, face_paths, metadata_path) -> None:
    faces_metadata = {}
    for video_path, label, group, face_paths_batch in zip(video_paths, labels, groups, face_paths):
        for face_path in face_paths_batch:
            faces_metadata[os.path.basename(face_path)] = {
                'label': int(label),
                'group': group,
                'video': os.path.basename(video_path)
            }

    with open(metadata_path, 'w') as outfile:
        json.dump(faces_metadata, outfile)


if __name__ == '__main__':
    np.random.seed(config.SEED)
    video_paths, video_file_names, labels, groups = get_train_data_lists()
    print('Total videos:', video_paths.shape)

    print('Extracting faces from videos ...')
    face_paths = extract_faces_from_videos_parallel(video_paths)

    print('Splitting data on train and validation ...')
    order_index = np.argsort(video_file_names)
    video_paths = video_paths[order_index]
    labels = labels[order_index]
    groups = groups[order_index]
    face_paths = np.sort(face_paths)

    face_paths = np.reshape(face_paths, (video_paths.shape[0], config.FRAMES_PER_VIDEO))

    gss = GroupShuffleSplit(n_splits=1, train_size=.95, random_state=config.SEED)
    train_idx, val_idx = next(gss.split(video_paths, labels, groups))
    print('Train len:', len(train_idx), ', Val len:', len(val_idx))

    print('Saving faces metadata ...')
    save_faces_metadata_to_json(
        video_paths[train_idx],
        labels[train_idx],
        groups[train_idx],
        face_paths[train_idx],
        config.FACES_TRAIN_METADATA_PATH)
    save_faces_metadata_to_json(
        video_paths[val_idx],
        labels[val_idx],
        groups[val_idx],
        face_paths[val_idx],
        config.FACES_VAL_METADATA_PATH)

    print('All done!')
