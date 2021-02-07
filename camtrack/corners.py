#! /usr/bin/env python3

__all__ = [
    'FrameCorners',
    'CornerStorage',
    'build',
    'dump',
    'load',
    'draw',
    'calc_track_interval_mappings',
    'calc_track_len_array_mapping',
    'without_short_tracks'
]

import click
import cv2
import numpy as np
import pims

from _corners import (
    FrameCorners,
    CornerStorage,
    StorageImpl,
    dump,
    load,
    draw,
    calc_track_interval_mappings,
    calc_track_len_array_mapping,
    without_short_tracks,
    create_cli
)


class _CornerStorageBuilder:

    def __init__(self, progress_indicator=None):
        self._progress_indicator = progress_indicator
        self._corners = dict()

    def set_corners_at_frame(self, frame, corners):
        self._corners[frame] = corners
        if self._progress_indicator is not None:
            self._progress_indicator.update(1)

    def build_corner_storage(self):
        return StorageImpl(item[1] for item in sorted(self._corners.items()))


def _build_impl(frame_sequence: pims.FramesSequence,
                builder: _CornerStorageBuilder) -> None:
    # https://docs.opencv.org/3.4/d4/dee/tutorial_optical_flow.html

    # params for ShiTomasi corner detection
    max_corners = 1000
    min_distance = 7
    feature_params = dict(minDistance=min_distance,
                          qualityLevel=0.01,
                          blockSize=7)

    # Parameters for lucas kanade optical flow
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
                     minEigThreshold=0.002)

    image_0 = frame_sequence[0]
    corner_points_0 = cv2.goodFeaturesToTrack(image_0, maxCorners=max_corners, mask=None, **feature_params)
    corner_points_0 = corner_points_0 if corner_points_0 is not None else np.zeros((0, 1, 2), dtype=np.float32)
    corner_points_0_ids = np.array(range(len(corner_points_0)))

    corners = FrameCorners(
        corner_points_0_ids,
        corner_points_0,
        np.full((len(corner_points_0)), feature_params['blockSize'])
    )
    builder.set_corners_at_frame(0, corners)

    all_corners_num = len(corner_points_0)
    for frame, image_1 in enumerate(frame_sequence[1:], 1):
        image_0_8bit = cv2.convertScaleAbs(image_0, alpha=255)
        image_1_8bit = cv2.convertScaleAbs(image_1, alpha=255)

        corner_points_1, st, err = cv2.calcOpticalFlowPyrLK(image_0_8bit, image_1_8bit,
                                                            corner_points_0, None, **lk_params)
        st = np.reshape(st, (-1))
        corner_points_1_ids = corner_points_0_ids[st == 1]
        corner_points_1 = corner_points_1[st == 1]

        # add new corners
        if len(corner_points_1) < max_corners:
            mask = np.ones_like(image_1, dtype=np.uint8)
            for ((x, y),) in corner_points_1:
                x, y = int(round(x)), int(round(y))
                mask = cv2.circle(mask, (x, y), radius=min_distance, color=0, thickness=cv2.FILLED)

            additional_corners_num = max_corners - len(corner_points_1)
            new_points = cv2.goodFeaturesToTrack(image_1, maxCorners=additional_corners_num, mask=mask, **feature_params)
            new_points = new_points if new_points is not None else np.zeros((0, 1, 2), dtype=np.float32)
            corner_points_1 = np.append(corner_points_1, new_points, axis=0)

            new_ids = np.array(range(all_corners_num, all_corners_num + len(new_points)), dtype=np.int32)
            all_corners_num += len(new_points)
            corner_points_1_ids = np.append(corner_points_1_ids, new_ids, axis=0)

        corners = FrameCorners(
            corner_points_1_ids,
            corner_points_1,
            np.full((len(corner_points_1)), feature_params['blockSize'])
        )

        builder.set_corners_at_frame(frame, corners)
        image_0 = image_1
        corner_points_0_ids = corner_points_1_ids
        corner_points_0 = corner_points_1


def build(frame_sequence: pims.FramesSequence,
          progress: bool = True) -> CornerStorage:
    """
    Build corners for all frames of a frame sequence.

    :param frame_sequence: grayscale float32 frame sequence.
    :param progress: enable/disable building progress bar.
    :return: corners for all frames of given sequence.
    """
    if progress:
        with click.progressbar(length=len(frame_sequence),
                               label='Calculating corners') as progress_bar:
            builder = _CornerStorageBuilder(progress_bar)
            _build_impl(frame_sequence, builder)
    else:
        builder = _CornerStorageBuilder()
        _build_impl(frame_sequence, builder)
    return builder.build_corner_storage()


if __name__ == '__main__':
    create_cli(build)()  # pylint:disable=no-value-for-parameter
