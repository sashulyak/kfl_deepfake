# Deepfake Detection Challenge Solution

[Deepfake Detection Challenge](https://www.kaggle.com/c/deepfake-detection-challenge/overview) was about identifying media manipulated with Deepfake techniques. The main goal is to create a binary classification model that predicts whether input video is fake or not.

## Data

Data represented with 4 datasets
- [Training Set](https://www.kaggle.com/c/deepfake-detection-challenge/data): dataset, containing labels for the target. The full training set is just over 470 GB. It is broken up into 50 groups ~10GB each.
- Public Validation Set: small dataset of 400 videos. This set is in use when you commit your Kaggle notebook. Accessible on the [same page](https://www.kaggle.com/c/deepfake-detection-challenge/data) as Training Set.
- Public Test Set: completely withheld and is what Kaggle’s platform computes the public leaderboard against.
- Private Test Set: privately held outside of Kaggle’s platform, and is used to compute the private leaderboard.

## Solution

Solution can be described with several steps.

### Data preparation
- Extract 17 faces from each video with MTCNN face detector
- Save each face on disk
- Save corresponded metadata (which face from which video)
- Divide training dataset on train and validation part

### Training
Train binary classificator based on Xception CNN.

### Validation and inference
- Extract 17 faces from each video
- Predict label [0..1] for each face separately using pretrained model from the previous step
- Use mean value of 17 predictions to get prediction whether input video is fake or not

## TODO:
- Migrate from Keras.Sequential to Tensorflow.data.Dataset
- Find working Tensorflow implementation of MTCNN and make it work with TF 2.0
- Clear data preparation and evaluation stages