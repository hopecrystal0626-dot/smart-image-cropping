# crop/bbox_utils.py

from dataclasses import dataclass


@dataclass
class BBox:
    """
    Bounding Box

    (x1, y1)
        ┌─────────┐
        │         │
        │         │
        └─────────┘
                 (x2, y2)
    """

    x1: int
    y1: int
    x2: int
    y2: int
    scale: float

    @property
    def width(self):
        return self.x2 - self.x1

    @property
    def height(self):
        return self.y2 - self.y1

    @property
    def area(self):
        return self.width * self.height

    @property
    def center_x(self):
        return (self.x1 + self.x2) / 2

    @property
    def center_y(self):
        return (self.y1 + self.y2) / 2

    @property
    def center(self):
        return (
            self.center_x,
            self.center_y
        )

    def to_tuple(self):
        """
        转换为(x1,y1,x2,y2)
        """
        return (
            self.x1,
            self.y1,
            self.x2,
            self.y2
        )

    def __str__(self):
        return (
            f"BBox("
            f"x1={self.x1}, "
            f"y1={self.y1}, "
            f"x2={self.x2}, "
            f"y2={self.y2})"
        )

def clip_bbox(bbox, img_w, img_h):
    """
    防止越界
    """

    return BBox(
        max(0, bbox.x1),
        max(0, bbox.y1),
        min(img_w, bbox.x2),
        min(img_h, bbox.y2)
    )
    
    
def compute_iou(box1, box2):
    """
    IoU
    """

    inter_x1 = max(box1.x1, box2.x1)
    inter_y1 = max(box1.y1, box2.y1)

    inter_x2 = min(box1.x2, box2.x2)
    inter_y2 = min(box1.y2, box2.y2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)

    inter_area = inter_w * inter_h

    union_area = (
        box1.area
        + box2.area
        - inter_area
    )

    if union_area == 0:
        return 0.0


def distance_between_centers(box1, box2):
    """
    两个框中心点距离
    """

    dx = box1.center_x - box2.center_x
    dy = box1.center_y - box2.center_y

    return (dx ** 2 + dy ** 2) ** 0.5