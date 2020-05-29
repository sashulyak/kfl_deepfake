from typing import List

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

        self.crop = ops.Crop(
            device="gpu",
            crop=(config.IMG_SIZE, config.IMG_SIZE),
            crop_pos_x=0,
            crop_pos_y=0
        )

    def define_graph(self):
        frames = self.input(name="Reader")
        output = self.crop(frames)
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
    #video_pipeline.build()

    features_dataset = dali_tf.DALIDataset(
        pipeline=video_pipeline,
        batch_size=1,
        output_shapes=(config.FRAMES_PER_VIDEO, config.IMG_SIZE, config.IMG_SIZE, 3),
        output_dtypes=tf.uint8,
        device_id=0
    )

    features_dataset_unbatched = features_dataset
    labels_dataset = tf.data.Dataset.from_tensor_slices(tf.constant(labels))

    return tf.data.Dataset.zip((features_dataset_unbatched, labels_dataset))
