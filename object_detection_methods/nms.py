import iou


def nms(dets, nms_threshold=0.5):
    # dets = [ [x1, y1, x2, y2, score], ...]
    # Sort detections by confidence score
    sorted_dets = sorted(dets, key=lambda k: -k[-1])

    # List of detections that we will return
    keep_dets = []
    while len(sorted_dets) > 0:
        keep_dets.append(sorted_dets[0])
        # Remove highest confidence box
        # and remove all boxes that have high overlap with it
        sorted_dets = [
            box
            for box in sorted_dets[1:]
            if iou.iou(sorted_dets[0][:-1], box[:-1]) < nms_threshold
        ]
    return keep_dets
