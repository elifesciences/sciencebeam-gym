from typing import NamedTuple


class BoundingRange(NamedTuple):
    start: float
    length: float

    def validate(self) -> 'BoundingRange':
        if self.length < 0:
            raise ValueError(f'length must not be less than zero, was: {self.length}')
        return self

    def __len__(self):
        return self.length

    def empty(self):
        return self.length == 0

    def intersects(self, other):
        return (
            (self.start < other.start + other.length) and
            (other.start < self.start + self.length)
        )

    def intersection(self, other: 'BoundingRange') -> 'BoundingRange':
        intersection_start = max(self.start, other.start)
        intersection_end = min(self.start + self.length, other.start + other.length)
        return BoundingRange(
            intersection_start,
            max(0, intersection_end - intersection_start)
        )

    def include(self, other):
        if other.empty():
            return self
        if self.empty():
            return other
        start = min(self.start, other.start)
        length = max(self.start + self.length, other.start + other.length) - start
        return BoundingRange(start, length).validate()

    def __add__(self, other):
        return self.include(other)


class BoundingBox(NamedTuple):
    x: float
    y: float
    width: float
    height: float

    def validate(self) -> 'BoundingBox':
        if self.width < 0:
            raise ValueError(f'width must not be less than zero, was: {self.width}')
        if self.height < 0:
            raise ValueError(f'height must not be less than zero, was: {self.height}')
        return self

    def __str__(self):
        return '({}, {}, {}, {})'.format(self.x, self.y, self.width, self.height)

    def __repr__(self):
        return 'BB({}, {}, {}, {})'.format(self.x, self.y, self.width, self.height)

    def __bool__(self) -> bool:
        return not self.empty()

    def to_list(self):
        return [self.x, self.y, self.width, self.height]

    def empty(self) -> bool:
        return self.width == 0 or self.height == 0

    def round(self) -> 'BoundingBox':
        return BoundingBox(int(self.x), int(self.y), int(self.width), int(self.height))

    def move_by(self, rx, ry):
        return BoundingBox(self.x + rx, self.y + ry, self.width, self.height)

    def scale_by(self, rx, ry):
        return BoundingBox(self.x * rx, self.y * ry, self.width * rx, self.height * ry)

    def expand_by(self, dx: float, dy: float) -> 'BoundingBox':
        return self.shrink_by(-dx, -dy)

    def shrink_by(self, dx: float, dy: float) -> 'BoundingBox':
        return BoundingBox(
            self.x + dx,
            self.y + dy,
            max(0, self.width - 2 * dx),
            max(0, self.height - 2 * dy)
        )

    def include(self, other):
        if other.empty():
            return self
        if self.empty():
            return other
        x = min(self.x, other.x)
        y = min(self.y, other.y)
        w = max(self.x + self.width, other.x + other.width) - x
        h = max(self.y + self.height, other.y + other.height) - y
        return BoundingBox(x, y, w, h)

    def with_margin(self, dx, dy=None):
        if dy is None:
            dy = dx
        return BoundingBox(
            self.x - dx,
            self.y - dy,
            self.width + 2 * dx,
            self.height + 2 * dy
        )

    def intersects(self, other):
        return (
            self.x_range().intersects(other.x_range()) and
            self.y_range().intersects(other.y_range())
        )

    def intersection(self, other: 'BoundingBox') -> 'BoundingBox':
        intersection_x_range = self.x_range().intersection(
            other.x_range()
        )
        intersection_y_range = self.y_range().intersection(
            other.y_range()
        )
        return BoundingBox(
            intersection_x_range.start,
            intersection_y_range.start,
            intersection_x_range.length,
            intersection_y_range.length
        )

    def __add__(self, bb):
        return self.include(bb)

    def x_range(self):
        return BoundingRange(self.x, self.width).validate()

    def y_range(self):
        return BoundingRange(self.y, self.height).validate()

    def __eq__(self, other):
        if other is None:
            return False
        return super().__eq__(other)


EMPTY_BOUNDING_BOX = BoundingBox(0, 0, 0, 0)
