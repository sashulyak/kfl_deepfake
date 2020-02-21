import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
TEST_VIDEOS_DIR = os.path.join(BASE_DIR, 'data/test_videos')
TRAIN_VIDEOS_DIR = os.path.join(BASE_DIR, 'data/train_videos')
SAMPLE_SUBMISSION_FILE = os.path.join(BASE_DIR, 'data/sample_submission.csv')
TRAIN_FACES_DIR = os.path.join(BASE_DIR, 'data/train_faces_mtcnn')
TRAIN_KEYPOINTS_DIR = os.path.join(BASE_DIR, 'data/train_keypoints_mtcnn')
FACES_TRAIN_METADATA_PATH = os.path.join(BASE_DIR, 'data/train_faces_mtcnn_metadata.json')
FACES_VAL_METADATA_PATH = os.path.join(BASE_DIR, 'data/val_faces_mtcnn_metadata.json')
SEED = 42
FRAMES_PER_VIDEO = 17

MODEL_PATH = os.path.join(BASE_DIR, 'models/deepfake_model.h5')
LOG_PATH = os.path.join(BASE_DIR, 'train_log.csv')
EPOCHS = 100
IMG_SIZE = 200
BATCH_SIZE = 192
VAL_BATCH_SIZE = 64
TRAIN_FAKE_RATIO = 0.8354250485398026