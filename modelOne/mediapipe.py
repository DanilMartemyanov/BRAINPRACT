

import cv2
import mediapipe as mp



# IMAGE_FILES = ['images/1 (1).jpg']


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

def mediapipe_for_images(file):
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5) as hands:

        image = cv2.imread(file)
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        return results.multi_handedness