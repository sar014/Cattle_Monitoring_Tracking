import numpy as np

def calculate_features(kpts, prev_center, current_center):
    """
    Calculates the 5 features your RF model expects.
    KEYPOINT_ORDER: [right-ear, nose, left-ear, neck, left-front-hoof, 
                     right-front-hoof, hip, left-back-hoof, right-back-hoof, tail]
    """
    # kpts is [10, 2] array
    # 1. nose_neck_y_diff (Nose is index 1, Neck is index 3)
    nose_neck_y_diff = kpts[1][1] - kpts[3][1]
    
    # 2. neck_hip_y_diff (Neck is 3, Hip is 6)
    neck_hip_y_diff = kpts[3][1] - kpts[6][1]
    
    # 3. movement (Euclidean distance between center of bbox in frame t and t-1)
    movement = 0
    if prev_center is not None:
        movement = np.linalg.norm(current_center - prev_center)
        
    # 4. tail_hip_dist (Tail is 9, Hip is 6)
    tail_hip_dist = np.linalg.norm(kpts[9] - kpts[6])
    
    # 5. stride_length (Using front hoof distance as a proxy)
    stride_length = np.linalg.norm(kpts[4] - kpts[5])

    return [nose_neck_y_diff, neck_hip_y_diff, stride_length, tail_hip_dist, movement]