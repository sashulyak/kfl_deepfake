from typing import List

import nvidia.dali.plugin.tf as dali_tf
import tensorflow as tf
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops

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

    def define_graph(self):
        output = self.input(name="Reader")
        return output


def get_dataset(video_file_paths: List[str], labels: List[int]) -> tf.Dataset:
    """
    Create Tensorflow dataset with direct video reader.

    :param video_file_paths: source video paths
    :param labels: corresponding videl labels
    :return: Tensorflow dataset of pairs (video frames, label)
    """
    video_pipeline = VideoPipeline(
        batch_size=config.BATCH_SIZE,
        num_threads=2,
        device_id=0,
        sequence_length=config.FRAMES_PER_VIDEO,
        initial_prefetch_size=16,
        data=video_file_paths,
        shuffle=True
    )
    video_pipeline.build()

    shapes = [(config.BATCH_SIZE, config.FRAMES_PER_VIDEO, config.IMG_SIZE, config.IMG_SIZE)]
    dtypes = [tf.float32]

    features_dataset = dali_tf.DALIDataset(
        pipeline=video_pipeline,
        batch_size=config.BATCH_SIZE,
        shapes=shapes,
        dtypes=dtypes,
        device_id=0
    )

    labels_dataset = tf.Dataset.from_tensor_slices(tf.constant(labels))

    return tf.Dataset.zip((features_dataset, labels_dataset))
