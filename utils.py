import numpy as np
from dataclasses import dataclass
import numpy_groupies as npg
import numpy.typing as npt


@dataclass
class Detections:
    frame_id: npt.NDArray[int]
    cls: npt.NDArray[str]
    x: npt.NDArray[float]
    y: npt.NDArray[float]
    width: npt.NDArray[float]
    height: npt.NDArray[float]
    track_id: npt.NDArray[int]

    def __post_init__(self):
        assert len(set([len(i) for k, i in self.__dict__.items() if k[:1] != '_'])) == 1


def is_in_dynamic_roi(detection: Detections, dynamic_roi_class: str) -> bool:
    """
    on each frame looking for a dynamic_roi_class-objects and if it present check which object on this frame is
    located inside this object/objects (center of this object is located inside dynamic_roi_class-objects)
    :return: mask for each detection if it located or not inside some of dynamic_roi_class-objects
    """

    def get_index_withing_repeatable(arr: npt.NDArray) -> npt.NDArray:
        """
        return indexes withing repeated elements indexes. E.g. sections of same values will be substituted with indexes
        of elements with this section.
        Example [0, 0, 0, 1, 1] -> [0, 1, 2, 0, 1]
        @param arr: should be sorted (e.g. frame_id)
        @return:
        """
        _, index, invers = np.unique(arr, return_inverse=True, return_index=True)
        return np.arange(len(arr)) - index[invers]

    def repeat_repeatable(arr: npt.NDArray) -> npt.NDArray:
        """
        repeated elements will be duplicated by amount of it number with association of each duplication with index
        example [0, 0, 0, 1, 1] -> [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4]
        @param arr:
        @return:
        """
        _, invers, count = np.unique(arr, return_inverse=True, return_counts=True)
        return np.repeat(np.arange(len(arr)), count[invers])

    def index_ordered_row(arr: npt.NDArray) -> npt.NDArray:
        """
        rendering result of repeat_repeatable()
        example [0, 0, 0, 1, 1] -> [0, 1, 2, 0, 1, 2, 0, 1, 2, 3, 4, 3, 4]
        @param arr:
        @return:
        """
        index_of_coords_to_repeat = repeat_repeatable(arr)
        index_within_unique = get_index_withing_repeatable(index_of_coords_to_repeat)
        _, index, count = np.unique(arr, return_counts=True, return_index=True)
        return np.repeat(index, count ** 2) + index_within_unique

    frame_id = detection.frame_id
    cls = detection.cls
    coords = np.stack(
        (
            detection.x,
            detection.y,
            detection.x + detection.width,
            detection.y + detection.height,
        ),
        axis=1,
    )

    ##placeholder
    max_cart_on_frame = 5  # max number of dynamic ROIs on single frame
    rois = np.zeros((len(frame_id), max_cart_on_frame, 4)) + (1, 1, -1, -1)  # (frame, cart_withing_frame, coords)

    ##index magic
    index_of_coords_to_repeat = repeat_repeatable(frame_id)
    index_within_unique = get_index_withing_repeatable(index_of_coords_to_repeat)
    ordered_index = index_ordered_row(frame_id)
    cart_mask = cls[ordered_index] == dynamic_roi_class
    rois[index_of_coords_to_repeat[cart_mask], index_within_unique[cart_mask], :] = coords[ordered_index[cart_mask]]

    roi_x_min, roi_y_min, roi_x_max, roi_y_max = rois[..., 0], rois[..., 1], rois[..., 2], rois[..., 3]
    x_center = detection.x + detection.width / 2
    y_center = detection.y + detection.height / 2
    res = np.any(
        (roi_x_min.T <= x_center) & (roi_x_max.T >= x_center) & (roi_y_min.T <= y_center) & (roi_y_max.T >= y_center),
        axis=0,
    ).T
    return res


def count_frames(detections: Detections, dynamic_roi_class: str):
    tracklet_hits = is_in_dynamic_roi(detections, dynamic_roi_class=dynamic_roi_class)
    track_ids=detections.track_id
    u_track_ids, inverse_indices = np.unique(track_ids, return_inverse=True)
    counts = npg.aggregate(group_idx=inverse_indices, a=tracklet_hits, func="sum")
    return dict(zip(u_track_ids, counts))
