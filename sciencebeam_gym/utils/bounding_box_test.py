from sciencebeam_gym.utils.bounding_box import (
  BoundingBox
)

class TestBoundingBox(object):
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
    assert BoundingBox(11, 12, 101, 102) != None

  def test_should_not_equal_to_none(self):
    assert None != BoundingBox(11, 12, 101, 102)  # pylint: disable=misplaced-comparison-constant

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
