"""
Estimación de pose (cuerpo + manos + cara opcional) con MediaPipe Holistic (0.10.x).
API 'solutions' clásica — estable en Apple Silicon (la Tasks API 1.x crashea en Metal).
"""
import cv2
import mediapipe as mp

mp_holistic = mp.solutions.holistic
mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles


class PoseEstimator:
    """Envuelve un Holistic. Reusar 1 por job (no es thread-safe)."""

    def __init__(self, complexity: int = 1, min_conf: float = 0.5):
        self.h = mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=int(complexity),   # 0 rápido, 1 balance, 2 preciso
            smooth_landmarks=True,
            refine_face_landmarks=False,
            min_detection_confidence=min_conf,
            min_tracking_confidence=min_conf,
        )

    def detect(self, rgb_frame):
        """rgb_frame: np.uint8 HxWx3 RGB. Devuelve los landmarks (o None)."""
        res = self.h.process(rgb_frame)
        return (
            res.pose_landmarks,
            res.left_hand_landmarks,
            res.right_hand_landmarks,
            res.face_landmarks,
        )

    def close(self):
        self.h.close()


def draw(image_bgr, landmarks, draw_face: bool = False):
    """Dibuja el esqueleto (cuerpo + manos [+ cara]) sobre una imagen BGR, in place."""
    pose_lm, lh_lm, rh_lm, face_lm = landmarks

    if draw_face and face_lm is not None:
        mp_draw.draw_landmarks(
            image_bgr, face_lm, mp_holistic.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_styles.get_default_face_mesh_contours_style(),
        )

    if pose_lm is not None:
        mp_draw.draw_landmarks(
            image_bgr, pose_lm, mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style(),
        )

    for hand_lm in (lh_lm, rh_lm):
        if hand_lm is not None:
            mp_draw.draw_landmarks(
                image_bgr, hand_lm, mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_styles.get_default_hand_landmarks_style(),
                connection_drawing_spec=mp_styles.get_default_hand_connections_style(),
            )
    return image_bgr
