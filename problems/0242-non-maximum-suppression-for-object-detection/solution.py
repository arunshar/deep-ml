import numpy as np

def non_maximum_suppression(boxes, scores, iou_threshold):
    # 1. Handle empty inputs gracefully
    if not boxes or not scores:
        return []
        
    # 2. Validate types and format
    try:
        boxes = np.array(boxes, dtype=np.float32)
        scores = np.array(scores, dtype=np.float32)
    except Exception:
        return -1
        
    # 3. Check shapes and bounds
    if boxes.ndim != 2 or boxes.shape[1] != 4 or scores.ndim != 1:
        return -1
    if len(boxes) != len(scores):
        return -1
    if not (0 <= iou_threshold <= 1):
        return -1
        
    # 4. Sort indices by score in descending order
    # Convert to a standard Python list so we can easily pop elements
    order = scores.argsort()[::-1].tolist()
    keep = []
    
    # 5. Iteratively filter boxes
    while order:
        # Take and remove the index of the box with the highest score
        i = order.pop(0)
        keep.append(int(i))
        
        # Build a new lis