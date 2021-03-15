#! /usr/bin/env python3

__all__ = [
    'track_and_calc_colors'
]

from typing import List, Optional, Tuple

import numpy as np

from corners import CornerStorage, FrameCorners, calc_track_interval_mappings
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
from collections import namedtuple


def build_3d_points(corners_1: FrameCorners, view_mat_1: np.ndarray,
                    corners_2: FrameCorners, view_mat_2: np.ndarray,
                    intrinsic_mat: np.ndarray,
                    start=False, ids_to_remove=None):
    if start:
        triangulation_parameters = TriangulationParameters(
            max_reprojection_error=3,
            min_triangulation_angle_deg=1,
            min_depth=0.5
        )
    else:
        triangulation_parameters = TriangulationParameters(
            max_reprojection_error=3,
            min_triangulation_angle_deg=5,
            min_depth=5
        )

    correspondences = build_correspondences(corners_1, corners_2, ids_to_remove)
    if len(correspondences.ids) == 0:
        return np.empty(shape=(0, 3), dtype=np.float32), np.empty(shape=(0,), dtype=np.int64)
    points_3d, corner_ids, _ = triangulate_correspondences(correspondences, view_mat_1, view_mat_2,
                                                           intrinsic_mat, triangulation_parameters)
    return points_3d, corner_ids


def determine_camera_pos(points_3d, points_2d, intrinsic_mat):
    try:
        if len(points_3d) < 4:
            return None, None
        retval, _, _, inliers = cv2.solvePnPRansac(points_3d, points_2d, intrinsic_mat, None)
        if not retval:
            return None, None
        retval, rvec, tvec = cv2.solvePnP(points_3d[inliers], points_2d[inliers],
                                          intrinsic_mat, None, flags=cv2.SOLVEPNP_ITERATIVE)
        if not retval:
            return None, None
        return rodrigues_and_translation_to_view_mat3x4(rvec, tvec), inliers
    except cv2.error as e:
        print(f'unexpected error in opencv: {e.msg}')
        return None, None


def fix_undetermined_frames(view_mats: List[Optional[np.ndarray]]) -> List[np.ndarray]:
    print(f"final undetermined frames number: {len([a for a in view_mats if a is None])}")
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
    rgb_sequence = frameseq.read_rgb_f32(frame_sequence_path)
    intrinsic_mat = to_opencv_camera_mat3x3(
        camera_parameters,
        rgb_sequence[0].shape[0]
    )

    if known_view_1 is None or known_view_2 is None:
        known_view_1, known_view_2 = init_frames(corner_storage, intrinsic_mat)

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

        # process frames that are close to frames with known position first
        frame_processing_order = sorted(list(unknown_positions),
                                        key=lambda idx: min([abs(idx - i) for i in known_positions]))
        for idx in frame_processing_order:
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


BestPositions = namedtuple('BestPositions', ('frame_1', 'frame_1_pose', 'frame_2', 'frame_2_pose', 'score'))


def init_frames(corner_storage: CornerStorage,
                intrinsic_mat: np.ndarray) -> Tuple[Tuple[int, Pose], Tuple[int, Pose]]:
    best_positions: Optional[BestPositions] = None
    corners_left_frame, corners_right_frame = calc_track_interval_mappings(corner_storage)
    all_frame_num = len(corner_storage)
    first_frame_num = 5
    first_frame_candidates = [all_frame_num // (first_frame_num - 1) * i for i in range(first_frame_num - 1)] + \
                             [all_frame_num - 1] + [1] # add second frame also
    for first_frame in first_frame_candidates:
        print('trying to find pair for frame ', first_frame)
        first_frame_corners = corner_storage[first_frame].ids
        if len(first_frame_corners) < 5:
            continue
        leftmost_frame = sorted(corners_left_frame[first_frame_corners].flatten())[4]
        rightmost_frame = sorted(corners_right_frame[first_frame_corners].flatten(), key=lambda x: -x)[4]
        if leftmost_frame >= rightmost_frame:
            continue
        second_frame_candidates = range(leftmost_frame, rightmost_frame + 1)
        for second_frame in second_frame_candidates:
            positions = get_possible_positions(first_frame, second_frame, corner_storage, intrinsic_mat)
            if positions is None:
                continue
            if best_positions is None or best_positions.score < positions.score:
                best_positions = positions
    if best_positions is None:
        raise Exception("Cannot choose initialization frames")
    print(f'initialization frames: {best_positions.frame_1}, {best_positions.frame_2}')
    return (best_positions.frame_1, best_positions.frame_1_pose), (best_positions.frame_2, best_positions.frame_2_pose)

def get_possible_positions(frame_1: int,
                           frame_2: int,
                           corner_storage: CornerStorage,
                           intrinsic_mat: np.ndarray) -> Optional[BestPositions]:
    correspondences = build_correspondences(corner_storage[frame_1], corner_storage[frame_2])
    if len(correspondences.ids) < 5:
        return None

    E, mask_essential = cv2.findEssentialMat(correspondences.points_1, correspondences.points_2, intrinsic_mat,
                                             method=cv2.RANSAC, prob=0.99, threshold=3.0)
    _, mask_homography = cv2.findHomography(correspondences.points_1, correspondences.points_2, method=cv2.RANSAC)

    essential_indices = np.nonzero(mask_essential.flatten())
    homography_indices = np.nonzero(mask_homography.flatten())
    essential_not_in_homography = np.setdiff1d(essential_indices, homography_indices)
    if len(essential_not_in_homography) < 5:
        return None

    # if len(essential_not_in_homography) / len(mask_essential) < 0.1:
    #     return None

    inliers, R, t, _ = cv2.recoverPose(E, correspondences.points_1, correspondences.points_2,
                                       intrinsic_mat, mask=mask_essential)

    view_mat1 = np.eye(3, 4)
    view_mat2 = np.hstack((R, t))
    triangulation_parameters = TriangulationParameters(
            max_reprojection_error=3,
            min_triangulation_angle_deg=1,
            min_depth=0.5
        )
    _, corner_ids, median_cos = triangulate_correspondences(correspondences,
                                                            view_mat1, view_mat2,
                                                            intrinsic_mat, triangulation_parameters)
    pose_1 = view_mat3x4_to_pose(view_mat1)
    pose_2 = view_mat3x4_to_pose(view_mat2)

    fraction_in_homography = len(essential_not_in_homography) / len(mask_essential)
    final_score = len(corner_ids) * fraction_in_homography
    return BestPositions(frame_1, pose_1, frame_2, pose_2, final_score)


if __name__ == '__main__':
    # pylint:disable=no-value-for-parameter
    create_cli(track_and_calc_colors)()
