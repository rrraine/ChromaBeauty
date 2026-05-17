import mediapipe as mp
import numpy as np
import cv2

# Landmark index groups (from MediaPipe face mesh topology)
LIPS_INDICES  = list(set([61,146,91,181,84,17,314,405,321,375,291,185,40,39,37,0,
                           267,269,270,409,306,292,308,415,310,311,312,13,82,81,80,191]))
CHEEK_L_IDX   = [116,117,118,119,120,100,142,203,206,207,216]
CHEEK_R_IDX   = [345,346,347,348,349,329,371,423,426,427,436]
EYELID_L_IDX  = [33,7,163,144,145,153,154,155,133]
EYELID_R_IDX  = [362,382,381,380,374,373,390,249,263]

mp_face_mesh = mp.solutions.face_mesh

def get_region_masks(image_rgb: np.ndarray) -> dict:
    """Returns dict of binary uint8 masks, same H×W as input."""
    H, W = image_rgb.shape[:2]
    masks = {k: np.zeros((H, W), dtype=np.uint8)
             for k in ("lips", "left_cheek", "right_cheek", "left_eye", "right_eye")}

    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as fm:
        res = fm.process(image_rgb)
        if not res.multi_face_landmarks:
            return masks

        lm = res.multi_face_landmarks[0].landmark
        def px(idx):
            return (int(lm[idx].x * W), int(lm[idx].y * H))

        for indices, key in [
            (LIPS_INDICES,  "lips"),
            (CHEEK_L_IDX,   "left_cheek"),
            (CHEEK_R_IDX,   "right_cheek"),
            (EYELID_L_IDX,  "left_eye"),
            (EYELID_R_IDX,  "right_eye"),
        ]:
            pts = np.array([px(i) for i in indices], dtype=np.int32)
            cv2.fillPoly(masks[key], [cv2.convexHull(pts)], 1)

    return masks