class BoundingRange(object):
  def __init__(self, start, length):
    self.start = start
    self.length = length
    if length < 0:
      raise ValueError('length must not be less than zero, was: ' + str(length))

  def __str__(self):
    return '({}, {})'.format(self.start, self.length)

  def __len__(self):
    return self.length

  def empty(self):
    return self.length == 0

  def intersects(self, other):
    return (self.start < other.start + other.length) and (other.start < self.start + self.length)

  def include(self, other):
    if other.empty():
      return self
    if self.empty():
      return other
    start = min(self.start, other.start)
    length = max(self.start + self.length, other.start + other.length) - start
    return BoundingRange(start, length)

  def __add__(self, other):
    return self.include(other)

class BoundingBox(object):
  def __init__(self, x, y, width, height):
    self.x = x
    self.y = y
    self.width = width
    self.height = height
    if width < 0:
      raise ValueError('width must not be less than zero, was: ' + str(width))
    if height < 0:
      raise ValueError('height must not be less than zero, was: ' + str(height))

  def __str__(self):
    return '({}, {}, {}, {})'.format(self.x, self.y, self.width, self.height)

  def __repr__(self):
    return 'BB({}, {}, {}, {})'.format(self.x, self.y, self.width, self.height)

  def empty(self):
    return self.width == 0 or self.height == 0

  def move_by(self, rx, ry):
    return BoundingBox(self.x + rx, self.y + ry, self.width, self.height)

  def scale_by(self, rx, ry):
    return BoundingBox(self.x * rx, self.y * ry, self.width * rx, self.height * ry)

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
    return self.x_range().intersects(other.x_range()) and self.y_range().intersects(other.y_range())

  def __add__(self, bb):
    return self.include(bb)

  def x_range(self):
    return BoundingRange(self.x, self.width)

  def y_range(self):
    return BoundingRange(self.y, self.height)

  def __eq__(self, other):
    return (
      other is not None and
      self.x == other.x and
      self.y == other.y and
      self.width == other.width and
      self.height == other.height
    )

  def __hash__(self):
    return hash((self.x, self.y, self.width, self.height))

BoundingBox.EMPTY = BoundingBox(0, 0, 0, 0)
