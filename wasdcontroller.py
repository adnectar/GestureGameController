import cv2
import mediapipe as mp
from pynput.keyboard import Controller as KeyboardController
from pynput.mouse import Controller as MouseController, Button
import time
import threading

keyboard = KeyboardController()
mouse = MouseController()
video_capture = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                       min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

current_keys = set()
gesture_cooldown = 0.8  # seconds
last_action_time = 0
lock = threading.Lock()  # for shared resources

def update_keys(new_keys):
    global current_keys
    for key in current_keys - new_keys:
        keyboard.release(key)
    for key in new_keys - current_keys:
        keyboard.press(key)
    current_keys = new_keys.copy()

def is_fist(landmarks):
    finger_tips = [
        mp_hands.HandLandmark.INDEX_FINGER_TIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
        mp_hands.HandLandmark.RING_FINGER_TIP,
        mp_hands.HandLandmark.PINKY_TIP,
    ]

    finger_pips = [
        mp_hands.HandLandmark.INDEX_FINGER_PIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
        mp_hands.HandLandmark.RING_FINGER_PIP,
        mp_hands.HandLandmark.PINKY_PIP,
    ]

    fingers_extended = 0
    for tip, pip in zip(finger_tips, finger_pips):
        if landmarks[tip].y < landmarks[pip].y:
            fingers_extended += 1

    return fingers_extended < 2

def count_extended_fingers(landmarks, hand_label):
    finger_tips_ids = [4, 8, 12, 16, 20]
    extended = []

    if hand_label == "Right":
        extended.append(landmarks[4].x < landmarks[3].x)
    else:
        extended.append(landmarks[4].x > landmarks[3].x)

    for tip_id in finger_tips_ids[1:]:
        extended.append(landmarks[tip_id].y < landmarks[tip_id - 2].y)
    return extended

def process_left_hand(landmarks, w, h, frame):
    active_keys = set()

    if is_fist(landmarks):
        # Approximate center of the fist
        fist_landmarks = [
            mp_hands.HandLandmark.WRIST,
            mp_hands.HandLandmark.INDEX_FINGER_MCP,
            mp_hands.HandLandmark.MIDDLE_FINGER_MCP,
            mp_hands.HandLandmark.RING_FINGER_MCP,
            mp_hands.HandLandmark.PINKY_MCP,
        ]

        cx = sum(landmarks[lm].x for lm in fist_landmarks) / len(fist_landmarks)
        cy = sum(landmarks[lm].y for lm in fist_landmarks) / len(fist_landmarks)

        wx = int(cx * w)
        wy = int(cy * h)

        for i in range(1, 3):
            cv2.line(frame, (i * w // 3, 0), (i * w // 3, h), (200, 200, 200), 2)
            cv2.line(frame, (0, i * h // 3), (w, i * h // 3), (200, 200, 200), 2)

        cv2.circle(frame, (wx, wy), 8, (0, 255, 255), -1)

        zone_x = wx * 3 // w
        zone_y = wy * 3 // h

        direction = (zone_x, zone_y)

        if direction == (1, 0): active_keys.add('w')
        elif direction == (0, 0): active_keys.update(['w', 'a'])
        elif direction == (2, 0): active_keys.update(['w', 'd'])
        elif direction == (0, 1): active_keys.add('a')
        elif direction == (2, 1): active_keys.add('d')
        elif direction == (0, 2): active_keys.update(['s', 'a'])
        elif direction == (1, 2): active_keys.add('s')
        elif direction == (2, 2): active_keys.update(['s', 'd'])

        cv2.putText(frame, f"Zone: ({zone_x}, {zone_y})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Keys: {', '.join(active_keys)}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Open Hand - Movement Paused", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    with lock:
        update_keys(active_keys)

def process_right_hand(landmarks, hand_label, frame):
    global last_action_time
    extended = count_extended_fingers(landmarks, hand_label)
    fingers_up = extended.count(True)
    now = time.time()

    if now - last_action_time > gesture_cooldown:
        if fingers_up == 5:
            keyboard.press(' ')
            keyboard.release(' ')
            cv2.putText(frame, "ROLL", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            last_action_time = now
        elif extended[1] and not any(extended[i] for i in [0, 2, 3, 4]):
            mouse.click(Button.left)
            cv2.putText(frame, "ATTACK", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            last_action_time = now
        elif extended[1] and extended[2] and not any(extended[i] for i in [0, 3, 4]):
            keyboard.press('r')
            keyboard.release('r')
            cv2.putText(frame, "HEAL", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            last_action_time = now
        else:
            cv2.putText(frame, "NEUTRAL", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    h, w, _ = frame.shape

    if results.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            hand_label = results.multi_handedness[i].classification[0].label
            landmarks = hand_landmarks.landmark
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            if hand_label == "Left":
                threading.Thread(target=process_left_hand, args=(landmarks, w, h, frame)).start()
            elif hand_label == "Right":
                threading.Thread(target=process_right_hand, args=(landmarks, hand_label, frame)).start()

    cv2.imshow('Zone-Based WASD Controller', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

update_keys(set())
video_capture.release()
cv2.destroyAllWindows()
