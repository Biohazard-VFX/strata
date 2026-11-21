import numpy as np
import pytest
from numpy.testing import assert_array_equal

from strata_training.data.sequences import clip_indices
from strata_training.eval.temporal import static_correspondences


def test_clips_do_not_cross_shots_gaps_or_holdout() -> None:
    entries = [{"shot": "a", "frame": i, "split": "train"} for i in (1, 2, 4, 5)]
    entries.append({"shot": "b", "frame": 1, "split": "test"})
    assert clip_indices(entries, 3) == [[0, 1], [2, 3], [4]]
    entries[1]["split"] = "test"
    with pytest.raises(ValueError, match="holdout"):
        clip_indices(entries, 2)


def test_static_tracks_reject_occlusion_and_preserve_pixel_coordinates() -> None:
    z = np.full((8, 8), 2.0)
    next_z = z.copy()
    next_z[0, 0] = 1
    intrinsics = np.array([[10.0, 0, 4], [0, 10, 4], [0, 0, 1]])
    y0, x0, y1, x1 = static_correspondences(z, next_z, intrinsics, intrinsics, np.eye(4), np.eye(4))
    assert_array_equal(y0, y1)
    assert_array_equal(x0, x1)
    assert len(y0) == 3
