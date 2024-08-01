import torch as th
import numpy as np

def apply_deltas(anchor, deltas):
    dx, dy, dw, dh = deltas

    x1, x2, y1, y2 = anchor

    width = x2- x1
    height = y2 - y1

    center_x = width / 2 + x1
    center_y = height / 2 + y1

    new_center_x = width * dx + center_x
    new_center_y = height * dy + center_y

    new_width = width * np.exp(dw)
    new_height = height * np.exp(dh)

    pred_x1 = new_center_x - 0.5 * new_width
    pred_y1 = new_center_y - 0.5 * new_height
    pred_x2 = new_center_x + 0.5 * new_width
    pred_y2 = new_center_y + 0.5 * new_height

    new_anchors = th.stack((pred_x1, pred_y1, pred_x2, pred_y2), dim = 1)

    return new_anchors


def iou(target_boxes, anchors):
    # Compute the amount of overlapping region over two boxes calculated as a percentage

    # Compute the left top most corner of the two
    # We will use th.max() for the top left by computing the (x, y) coordinates that are the largest between the two
    # x1, y1 is the top left coordinate
    # Remember that target_boxes takes the form of [N, 4] while anchors takes the form of [M, 4]
    # Therefore, we need to reshape the box size of the target boxes so that they can be broadcast among and generate M * N total top left and bottom right results
    top_left = th.max(target_boxes[:, None, :2], anchors[:, :2])
    bottom_right = th.min(target_boxes[:, None, 2:], anchors[:, 2:])

    width_height = (bottom_right - top_left).clamp(min = 0)

    iou_area = abs(width_height[:, :, 0]) * abs(width_height[:, : ,1])

    target_box_area = abs(target_boxes[:, 0] - target_boxes[:, 2]) * abs(target_boxes[:, 1] - target_boxes[:, 3])
    anchor_area = abs(anchors[:, 0] - anchors[:, 2]) * abs(anchors[:, 1] - anchors[:, 3])

    return iou_area / (target_box_area[:, None] - iou_area + anchor_area) # Again need to broadcast the target box area so that it can become M by N