from typing import List, Tuple

import nvidia.dali.plugin.tf as dali_tf
import tensorflow as tf
import nvidia.dali.ops as ops
from nvidia.dali.pipeline import Pipeline

import config


class VideoPipeline(Pipeline):
    """ Nvidia DALI video reader pipeline for using with Tensorflow Dataset."""

    def __init__(
            self,
            batch_size: int,
            num_threads: int,
            device_id: int,
            sequence_length: int,
            initial_prefetch_size: int,
            data: List,
            shuffle: bool
    ):
        """
        :param batch_size: size of dataset batch
        :param num_threads: number of parallel threads
        :param device_id: "gpu" or "cpu"
        :param sequence_length: number of frames
        :param initial_prefetch_size: count of videos readed preliminarily
        :param data: input video paths
        :param shuffle: suffle samples
        """
        super().__init__(batch_size, num_threads, device_id, seed=16)
        self.input = ops.VideoReader(
            device="gpu",
            filenames=data,
            sequence_length=sequence_length,
            shard_id=0,
            num_shards=1,
            random_shuffle=shuffle,
            initial_fill=initial_prefetch_size
        )

        self.extract = ops.ElementExtract(
            device="gpu",
            element_map=list(range(0, config.FRAMES_PER_VIDEO))
        )

        self.resize = ops.Resize(
            device="gpu",
            resize_x=config.IMG_SIZE,
            resize_y=config.IMG_SIZE
        )

    def define_graph(self):
        frames = self.input(name="Reader")
        extracted_frames = self.extract(frames)
        resized_frames = []
        for extracted_frame in extracted_frames:
            resized_frames.append(self.resize(extracted_frame))

        return resized_frames[0], resized_frames[1], resized_frames[2]


# def unwrap_frames_sequence(
#     frames: tf.Tensor,
#     label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
#     """
#     Unwrap sequence into separate frames with labels.

#     :param ...: path to face crop file
#     :param label: corresponded label Tensor
#     :return: pair of preprocessed image Tensor and corresponded label Tensor
#     """
#     return tf.data.Dataset.from_tensor_slices([frames[0], frames[1], frames[2]]), tf.data.Dataset.from_tensor_slices([label, label, label])


def unwrap_frames_sequence(frames: tf.Tensor) -> tf.data.Dataset:
    """
    Unwrap sequence into separate frames with labels.

    :param ...: path to face crop file
    :param label: corresponded label Tensor
    :return: pair of preprocessed image Tensor and corresponded label Tensor
    """
    #return tf.data.Dataset.from_tensor_slices(frames)
    return tf.stack(frames)


def get_dali_dataset(video_file_paths: List[str], labels: List[int]) -> tf.data.Dataset:
    """
    Create Tensorflow dataset with direct video reader.

    :param video_file_paths: source video paths
    :param labels: corresponding videl labels
    :return: Tensorflow dataset of pairs (video frames, label)
    """
    video_pipeline = VideoPipeline(
        batch_size=1,
        num_threads=2,
        device_id=0,
        sequence_length=config.FRAMES_PER_VIDEO,
        initial_prefetch_size=16,
        data=video_file_paths,
        shuffle=True
    )

    features_dataset = dali_tf.DALIDataset(
        pipeline=video_pipeline,
        batch_size=1,
        output_shapes=(config.IMG_SIZE, config.IMG_SIZE, 3),
        output_dtypes=tf.uint8,
        device_id=0
    )

    features_dataset = features_dataset.map(unwrap_frames_sequence)

    labels_tensor = tf.constant(labels)
    labels_tensor = tf.tile(
        labels_tensor,
        tf.constant([config.FRAMES_PER_VIDEO], tf.int32)
    )
    labels_dataset = tf.data.Dataset.from_tensor_slices(labels_tensor)

    dataset = tf.data.Dataset.zip((features_dataset, labels_dataset))

    return dataset
