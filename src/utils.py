from typing import List

import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN

import config

# import keras.backend.tensorflow_backend as tb
# tb._SYMBOLIC_SCOPE.value = True


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

    return int(start_x_result), int(start_y_result), int(end_x_result), int(end_y_result)


def read_faces_from_video(detector: MTCNN, path: str, img_size=None, swap_channels=True) -> List[str]:
    capture = cv2.VideoCapture(path)
    num_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    video_faces = []
    video_keypoints = []
    for i in range(0, num_frames):
        ret = capture.grab()
        if i % 10 == 0:
            ret, frame = capture.retrieve()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = detector.detect_faces(frame_rgb)
            boxes = [face['box'] for face in faces]
            confidences = [face['confidence'] for face in faces]
            keypoints = [face['keypoints'] for face in faces]
            if len(faces) > 0:
                most_confident_face_index = np.argmax(confidences)
                (start_x, start_y, width, height) = boxes[most_confident_face_index]
                end_x = start_x + width
                end_y = start_y + height
                (start_x, start_y, end_x, end_y) = extend_rect_to_square(
                    start_x,
                    start_y,
                    end_x,
                    end_y,
                    frame.shape[1],
                    frame.shape[0])
                if swap_channels:
                    face_crop = frame_rgb[start_y:end_y, start_x:end_x]
                else:
                    face_crop = frame[start_y:end_y, start_x:end_x]
                if face_crop.shape[0] > 0 and face_crop.shape[1] > 0:
                    if img_size:
                        face_crop = cv2.resize(face_crop, (img_size, img_size))
                    video_faces.append(face_crop)
                    video_keypoints.append({
                        'box': (start_x, start_y, end_x, end_y),
                        'keypoints': keypoints[most_confident_face_index]
                    })
            if len(video_faces) == config.FRAMES_PER_VIDEO:
                break
    capture.release()
    assert len(video_faces) > 0
    return video_faces, video_keypoints
