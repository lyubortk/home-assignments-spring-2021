#! /usr/bin/env python3

__all__ = [
    'track_and_calc_colors'
]

from typing import List, Optional, Tuple

import numpy as np

from corners import CornerStorage, FrameCorners
from data3d import CameraParameters, PointCloud, Pose
import frameseq
from _camtrack import (
    PointCloudBuilder,
    create_cli,
    calc_point_cloud_colors,
    pose_to_view_mat3x4,
    to_opencv_camera_mat3x3,
    view_mat3x4_to_pose,
    TriangulationParameters,
    build_correspondences,
    triangulate_correspondences,
    rodrigues_and_translation_to_view_mat3x4
)
import cv2


def build_3d_points(corners_1: FrameCorners, view_mat_1: np.ndarray,
                    corners_2: FrameCorners, view_mat_2: np.ndarray,
                    intrinsic_mat: np.ndarray,
                    start=False, ids_to_remove=None):
    if start:
        triangulation_parameters = TriangulationParameters(
            max_reprojection_error=1,
            min_triangulation_angle_deg=1,
            min_depth=0.5
        )
    else:
        triangulation_parameters = TriangulationParameters(
            max_reprojection_error=0.2,
            min_triangulation_angle_deg=5,
            min_depth=5
        )

    correspondences = build_correspondences(corners_1, corners_2, ids_to_remove)
    points_3d, corner_ids, _ = triangulate_correspondences(correspondences, view_mat_1, view_mat_2,
                                                           intrinsic_mat, triangulation_parameters)
    return points_3d, corner_ids


def determine_camera_pos(points_3d, points_2d, intrinsic_mat):
    if len(points_3d) < 4:
        return None, None
    retval, rvec, tvec, inliers = cv2.solvePnPRansac(points_3d, points_2d, intrinsic_mat, None)
    if retval:
        return rodrigues_and_translation_to_view_mat3x4(rvec, tvec), inliers
    return None, None


def fix_undetermined_frames(view_mats: List[Optional[np.ndarray]]) -> List[np.ndarray]:
    view_mats = list(view_mats)
    frame_count = len(view_mats)

    frame_should_be_fixed = [a is None for a in view_mats]
    frame_fixed_by = [1000000000000000] * frame_count

    prev_determined_frame = None
    for i in range(frame_count):
        if not frame_should_be_fixed[i]:
            prev_determined_frame = i
            continue
        if prev_determined_frame is not None and abs(frame_fixed_by[i] - i) > abs(prev_determined_frame - i):
            frame_fixed_by[i] = prev_determined_frame
            view_mats[i] = view_mats[prev_determined_frame]

    prev_determined_frame = None
    for i in range(frame_count - 1, -1, -1):
        if not frame_should_be_fixed[i]:
            prev_determined_frame = i
            continue
        if prev_determined_frame is not None and abs(frame_fixed_by[i] - i) > abs(prev_determined_frame - i):
            frame_fixed_by[i] = prev_determined_frame
            view_mats[i] = view_mats[prev_determined_frame]

    return view_mats


def track_and_calc_colors(camera_parameters: CameraParameters,
                          corner_storage: CornerStorage,
                          frame_sequence_path: str,
                          known_view_1: Optional[Tuple[int, Pose]] = None,
                          known_view_2: Optional[Tuple[int, Pose]] = None) \
        -> Tuple[List[Pose], PointCloud]:
    if known_view_1 is None or known_view_2 is None:
        raise NotImplementedError()

    rgb_sequence = frameseq.read_rgb_f32(frame_sequence_path)
    intrinsic_mat = to_opencv_camera_mat3x3(
        camera_parameters,
        rgb_sequence[0].shape[0]
    )

    frame_count = len(corner_storage)
    view_mats: List[Optional[np.ndarray]] = [None] * frame_count
    view_mats[known_view_1[0]] = pose_to_view_mat3x4(known_view_1[1])
    view_mats[known_view_2[0]] = pose_to_view_mat3x4(known_view_2[1])

    points_3d, corner_ids = build_3d_points(corner_storage[known_view_1[0]], view_mats[known_view_1[0]],
                                            corner_storage[known_view_2[0]], view_mats[known_view_2[0]],
                                            intrinsic_mat, start=True)
    point_cloud_builder = PointCloudBuilder(corner_ids, points_3d)

    found_new_view_mats = True
    unknown_positions = set(range(frame_count))
    unknown_positions.remove(known_view_1[0])
    unknown_positions.remove(known_view_2[0])
    known_positions = {known_view_1[0], known_view_2[0]}

    while found_new_view_mats and len(unknown_positions) > 0:
        found_new_view_mats = False

        for idx in list(unknown_positions):
            print(f"processing frame {idx}\n"
                  f"point cloud size: {len(point_cloud_builder.ids)}\n"
                  f"undetermined frames {len(unknown_positions)}\n")
            point_ids, corner_mask, cloud_mask = np.intersect1d(corner_storage[idx].ids.flatten(),
                                                                point_cloud_builder.ids.flatten(),
                                                                return_indices=True)
            points_3d = point_cloud_builder.points[cloud_mask]
            points_2d = corner_storage[idx].points[corner_mask]
            new_view_mat, inliers = determine_camera_pos(points_3d, points_2d, intrinsic_mat)
            if new_view_mat is None:
                print(f"position for frame {idx} cannot (yet?) be determined")
                continue

            print(f"determined position for frame {idx} with {len(inliers)} inliers")
            outliers = np.delete(point_ids, inliers)
            found_new_view_mats = True
            view_mats[idx] = new_view_mat

            for known_position in known_positions:
                points_3d, corner_ids = build_3d_points(corner_storage[idx], view_mats[idx],
                                                        corner_storage[known_position], view_mats[known_position],
                                                        intrinsic_mat, ids_to_remove=outliers)
                point_cloud_builder.add_points(corner_ids, points_3d)
            known_positions.add(idx)
            unknown_positions.remove(idx)

    ## set undetermined view mats to the closest frame
    view_mats: List[np.ndarray] = fix_undetermined_frames(view_mats)

    calc_point_cloud_colors(
        point_cloud_builder,
        rgb_sequence,
        view_mats,
        intrinsic_mat,
        corner_storage,
        5.0
    )
    point_cloud = point_cloud_builder.build_point_cloud()
    poses = list(map(view_mat3x4_to_pose, view_mats))
    return poses, point_cloud


if __name__ == '__main__':
    # pylint:disable=no-value-for-parameter
    create_cli(track_and_calc_colors)()
