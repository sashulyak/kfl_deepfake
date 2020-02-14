import os
import json
from multiprocessing import Pool
from typing import List, Tuple


import cv2
import cvlib as cv
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from tqdm import tqdm


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
TEST_VIDEOS_DIR = os.path.join(BASE_DIR, 'data/test_videos')
TRAIN_VIDEOS_DIR = os.path.join(BASE_DIR, 'data/train_videos')
SAMPLE_SUBMISSION_FILE = os.path.join(BASE_DIR, 'data/sample_submission.csv')
TRAIN_FACES_DIR = os.path.join(BASE_DIR, 'data/train_faces')
FACES_TRAIN_METADATA_PATH = os.path.join(BASE_DIR, 'data/train_faces_metadata.json')
FACES_VAL_METADATA_PATH = os.path.join(BASE_DIR, 'data/val_faces_metadata.json')
SEED = 42
FRAMES_PER_VIDEO = 3

np.random.seed(SEED)


def get_train_data_lists() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    chunks_list = os.listdir(TRAIN_VIDEOS_DIR)

    paths = []
    labels = []
    groups = []
    names = []
    for chunk_dir_name in chunks_list:
        chunk_dir = os.path.join(TRAIN_VIDEOS_DIR, chunk_dir_name)
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


def get_image_name_from_video_name(video_name, frame_number=None):
    if frame_number is not None:
        return video_name[:-4] + '_{}.png'.format(frame_number)
    else:
        return video_name[:-4] + '.png'


def extend_rect_to_square(start_x, start_y, end_x, end_y, image_width, image_height):
    width = end_x - start_x
    height = end_y - start_y
    if width > height:
        difference = width - height
        start_y -= difference // 2
        end_y += difference // 2
    else:
        difference = height - width
        start_x -= difference // 2
        end_x += difference // 2
    start_x_result = np.max([0, start_x])
    start_y_result = np.max([0, start_y])
    end_x_result = np.min([image_width, end_x])
    end_y_result = np.min([image_height, end_y])

    return start_x_result, start_y_result, end_x_result, end_y_result


def save_faces_from_video(path: str) -> List[str]:
    video_name = os.path.basename(path)
    image_paths = []
    for i in range(FRAMES_PER_VIDEO):
        image_name = get_image_name_from_video_name(video_name, i)
        image_paths.append(os.path.join(TRAIN_FACES_DIR, image_name))

    for image_path in image_paths:
        if os.path.exists(image_path):
            return image_paths

    capture = cv2.VideoCapture(path)
    num_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    faces_to_save = []
    for i in range(0, num_frames):
        ret = capture.grab()
        if i % 10 == 0:
            ret, frame = capture.retrieve()
            faces, confidences = cv.detect_face(frame)
            if len(confidences) > 0:
                most_confident_face_index = np.argmax(confidences)
                (start_x, start_y, end_x, end_y) = faces[most_confident_face_index]
                (start_x, start_y, end_x, end_y) = extend_rect_to_square(
                    start_x,
                    start_y,
                    end_x,
                    end_y,
                    frame.shape[1],
                    frame.shape[0])
                face_crop = frame[start_y:end_y, start_x:end_x]
                if face_crop.shape[0] > 0 and face_crop.shape[1] > 0:
                    faces_to_save.append(face_crop)
            if len(faces_to_save) == FRAMES_PER_VIDEO:
                break
    capture.release()
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

    face_paths = np.reshape(face_paths, (video_paths.shape[0], FRAMES_PER_VIDEO))

    gss = GroupShuffleSplit(n_splits=1, train_size=.95, random_state=SEED)
    train_idx, val_idx = next(gss.split(video_paths, labels, groups))
    print("Train len:", len(train_idx), ", Val len:", len(val_idx))

    print('Saving faces metadata ...')
    save_faces_metadata_to_json(
        video_paths[train_idx],
        labels[train_idx],
        groups[train_idx],
        face_paths[train_idx],
        FACES_TRAIN_METADATA_PATH)
    save_faces_metadata_to_json(
        video_paths[val_idx],
        labels[val_idx],
        groups[val_idx],
        face_paths[val_idx],
        FACES_VAL_METADATA_PATH)

    print('All done!')
