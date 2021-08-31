import pytest

from sciencebeam_gym.utils.bounding_box import (
    BoundingBox
)


class TestBoundingBox(object):
    def test_should_accept_positive_width_and_height(self):
        bounding_box = BoundingBox(0, 0, 100, 100)
        assert bounding_box.validate() == bounding_box

    def test_should_accept_zero_width_and_height(self):
        bounding_box = BoundingBox(0, 0, 0, 0)
        assert bounding_box.validate() == bounding_box

    def test_should_reject_negative_width(self):
        with pytest.raises(ValueError):
            assert BoundingBox(0, 0, -100, 100).validate()

    def test_should_reject_negative_height(self):
        with pytest.raises(ValueError):
            assert BoundingBox(0, 0, 100, -100).validate()

    def test_should_indicate_empty_with_zero_width(self):
        assert BoundingBox(0, 0, 0, 100).empty()

    def test_should_indicate_empty_with_zero_height(self):
        assert BoundingBox(0, 0, 100, 0).empty()

    def test_should_indicate_not_be_empty_with_non_zero_width_and_height(self):
        assert not BoundingBox(0, 0, 100, 100).empty()

    def test_should_equal_same_bounding_boxes(self):
        assert BoundingBox(11, 12, 101, 102) == BoundingBox(11, 12, 101, 102)

    def test_should_not_equal_bounding_boxes_with_different_x(self):
        assert BoundingBox(11, 12, 101, 102) != BoundingBox(99, 12, 101, 102)

    def test_should_not_equal_bounding_boxes_with_different_y(self):
        assert BoundingBox(11, 12, 101, 102) != BoundingBox(11, 99, 101, 102)

    def test_should_not_equal_bounding_boxes_with_different_width(self):
        assert BoundingBox(11, 12, 101, 102) != BoundingBox(11, 12, 999, 102)

    def test_should_not_equal_bounding_boxes_with_different_height(self):
        assert BoundingBox(11, 12, 101, 102) != BoundingBox(11, 12, 101, 999)

    def test_should_not_equal_none(self):
        assert not BoundingBox(11, 12, 101, 102).__eq__(None)

    def test_should_include_another_bounding_box_to_the_bottom_right(self):
        assert (
            BoundingBox(10, 20, 50, 100).include(BoundingBox(100, 100, 200, 200)) ==
            BoundingBox(10, 20, 100 + 200 - 10, 100 + 200 - 20)
        )

    def test_should_include_another_bounding_box_to_the_top_left(self):
        assert (
            BoundingBox(100, 100, 200, 200).include(BoundingBox(10, 20, 50, 100)) ==
            BoundingBox(10, 20, 100 + 200 - 10, 100 + 200 - 20)
        )

    def test_should_calculate_intersection_with_identical_bounding_box(self):
        bounding_box = BoundingBox(110, 120, 50, 60)
        assert (
            bounding_box.intersection(bounding_box) == bounding_box
        )

    def test_should_calculate_intersection_with_smaller_contained_bounding_box(self):
        assert (
            BoundingBox(100, 100, 200, 200).intersection(
                BoundingBox(110, 120, 50, 60)
            ) == BoundingBox(110, 120, 50, 60)
        )

    def test_should_calculate_intersection_with_larger_bounding_box(self):
        assert (
            BoundingBox(110, 120, 50, 60).intersection(
                BoundingBox(100, 100, 200, 200)
            ) == BoundingBox(110, 120, 50, 60)
        )

    def test_should_calculate_intersection_with_overlapping_bounding_box(self):
        assert (
            BoundingBox(110, 120, 50, 60).intersection(
                BoundingBox(120, 110, 100, 100)
            ) == BoundingBox(120, 120, 40, 60)
        )

    def test_should_return_list(self):
        assert BoundingBox(11, 12, 101, 102).to_list() == [11, 12, 101, 102]
