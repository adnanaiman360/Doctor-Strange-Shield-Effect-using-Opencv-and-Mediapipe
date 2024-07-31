import math
import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
font = cv2.FONT_HERSHEY_SIMPLEX

# Function to check if the first two fingers and thumb are extended for the right hand
def is_three_fingers_extended(landmarks):
    thumb_is_open = landmarks[mp_hands.HandLandmark.THUMB_TIP].x > landmarks[mp_hands.HandLandmark.THUMB_IP].x
    index_is_open = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < landmarks[mp_hands.HandLandmark.INDEX_FINGER_PIP].y
    middle_is_open = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y < landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y
    ring_is_closed = landmarks[mp_hands.HandLandmark.RING_FINGER_TIP].y > landmarks[mp_hands.HandLandmark.RING_FINGER_PIP].y
    pinky_is_closed = landmarks[mp_hands.HandLandmark.PINKY_TIP].y > landmarks[mp_hands.HandLandmark.PINKY_PIP].y

    return thumb_is_open and index_is_open and middle_is_open and ring_is_closed and pinky_is_closed

def mapFromTo(x,a,b,c,d):
    return (x-a)/(b-a)*(d-c)+c

def Overlay (background, overlay, x, y, size):
    background_h, background_w, c = background.shape
    imgScale = mapFromTo(size, 200, 20, 1.5, 0.2)
    overlay = cv2.resize(overlay, (0, 0), fx=imgScale, fy=imgScale)
    h, w, c = overlay.shape
    try:
        if x + w/2 >= background_w or y + h/2 >= background_h or x - w/2 <= 0 or y - h/2 <= 0:
            return background
        else:
            overlayImage = overlay[..., :3]
            mask = overlay / 255.0
            background[int(y-h/2):int(y+h/2), int(x-w/2):int(x+w/2)] = (1-mask)*background[int(y-h/2):int(y+h/2), int(x-w/2):int(x+w/2)] + overlay
            return background
    except:
        return background

# Function to draw spark-like effects
def draw_spark_effect(image, points, shield_frame):
    for point in points:
        cx, cy = int(point.x * image.shape[1]), int(point.y * image.shape[0])
        handSize = 100  # You can adjust this size based on your needs
        image = Overlay(image, shield_frame, cx, cy, handSize-50)
    return image

# Webcam input
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
shield = cv2.VideoCapture("shield.mp4")

with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            success, shield_frame = shield.read()
            if not success:
                shield.set(cv2.CAP_PROP_POS_FRAMES, 0)
                success, shield_frame = shield.read()

            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

                if is_three_fingers_extended(hand_landmarks.landmark):
                    print("Three fingers detected.")
                    # Get the tips of the thumb, index, and middle fingers
                    points = [
                        # hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP],
                        # hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP],
                        hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
                    ]
                    image = draw_spark_effect(image, points, shield_frame)
                else:
                    print("Gesture not detected or not recognized.")

        cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
