from typing import List

import numpy as np
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

    def define_graph(self):
        output = self.input(name="Reader")
        return output


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
    # video_pipeline.build()

    shapes = (config.FRAMES_PER_VIDEO, config.IMG_SIZE, config.IMG_SIZE)
    dtypes = tf.float32

    features_dataset = dali_tf.DALIDataset(
        pipeline=video_pipeline,
        batch_size=1,
        output_shapes=shapes,
        output_dtypes=dtypes,
        device_id=0
    )

    features_dataset_unbatched = features_dataset.unbatch().unbatch()
    labels_dataset = tf.Dataset.from_tensor_slices(
        tf.repeat(labels, repeats=np.ones_like(labels) * config.FRAMES_PER_VIDEO, axis=0)
    )

    return tf.Dataset.zip((features_dataset_unbatched, labels_dataset))
