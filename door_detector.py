from ultralytics import YOLO
import cv2
import face_recognition
import numpy as np
import pygame
import time
import os
import sys

# ---------------------------------------------------------------------------
# Sound helpers
# ---------------------------------------------------------------------------

pygame.mixer.init()

sound_files = {
    "background": "toto_africa.mp3",
    "friend":     "me_and_your_mama.mp3.mp3",
    "thief":      "horror-tension-suspense-322304.mp3",
    "delivery":   "dr_alban.mp3"
}

COLOR = {
    "thief":     (0, 0, 255),
    "friend":    (0, 255, 0),
    "delivery":  (255, 255, 0),
    "noperson":  (255, 255, 255)
}

delays = {
    "friend":   0,
    "thief":    0,
    "delivery": 0,
    "noperson": 0
}

def play_sound(file, loop=0, start=0.0):
    """Load *file* and play it; loop=-1 means infinite looping."""
    if file != "toto_africa.mp3":
        pygame.mixer.music.load(file)
        pygame.mixer.music.play(loop, 27.0)    
    else:
        pygame.mixer.music.load(file)
        pygame.mixer.music.play(loop, start)

current_track = None  # keeps the key from sound_files of the track playing

def play_track(name, loop=0):
    """Convenience wrapper that stops whatever is playing first."""
    global current_track
    pygame.mixer.music.stop()
    play_sound(sound_files[name], loop)
    current_track = name

# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------
play_track("background", loop=-1)   # start background music

# YOLO model (make sure yolov8n.pt is present)
model = YOLO("yolov8n.pt")

# Face encodings for friends
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
friends_dir  = os.path.join(BASE_DIR, "friends")
known_face_encodings = []

if not os.path.isdir(friends_dir):
    print("ERROR: friends directory not found:", friends_dir)
    sys.exit(1)

for file in os.listdir(friends_dir):
    if file.lower().endswith((".png", ".jpg", ".jpeg")):
        img_path = os.path.join(friends_dir, file)
        image    = face_recognition.load_image_file(img_path)
        encs     = face_recognition.face_encodings(image)
        if encs:
            known_face_encodings.append(encs[0])

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: could not open webcam")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def classify_visitor(frame, results):
    """
    Return one of: 'friend', 'delivery', 'thief', 'noperson'
    """
    detected = []
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            detected.append(model.names[cls])

    # 1️⃣ No person → noperson
    if "person" not in detected:
        return "noperson"

    # 2️⃣ Friend?
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = face_recognition.face_locations(rgb, model="hog")
    encs  = face_recognition.face_encodings(rgb, faces)
    for enc in encs:
        if True in face_recognition.compare_faces(known_face_encodings, enc):
            return "friend"

    # 3️⃣ Delivery cues
    for label in detected:
        if label in ["handbag", "backpack", "helmet", "suit", "hat"]:
            return "delivery"

    # 4️⃣ Default
    return "thief"

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

last_event        = None
event_start_time  = 0
event_played      = False
BLINK_INTERVAL    = 0.5

def thief_blink(now):
    """True during ON half of blink cycle."""
    return (now % (2 * BLINK_INTERVAL)) < BLINK_INTERVAL

print("Press ESC to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    visitor = classify_visitor(frame, results)
    now     = time.time()

    # --- Overlay text -------------------------------------------------------
    if visitor == "thief":
        if thief_blink(now):
            cv2.putText(frame, "Visitor: thief", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR["thief"], 2)
    elif visitor == "noperson":
        cv2.putText(frame, "No person detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR["noperson"], 2)
    else:
        cv2.putText(frame, f"Visitor: {visitor}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR[visitor], 2)

    cv2.imshow("Webcam", frame)

    # --- Event handling -----------------------------------------------------
    if visitor != last_event:
        last_event        = visitor
        event_start_time  = now
        event_played      = False
        pygame.mixer.music.stop()  # free channel immediately

    if visitor == "noperson":
        # keep background looping
        if current_track != "background" or not pygame.mixer.music.get_busy():
            play_track("background", loop=-1)

    else:
        # friend / delivery / thief
        if not event_played and (now - event_start_time) >= delays[visitor]:
            play_track(visitor)   # one‑shot
            event_played = True

        # when one‑shot ends, resume background
        if event_played and not pygame.mixer.music.get_busy():
            play_track("background", loop=-1)
            event_played = False
            event_start_time = now

    # --- Quit on ESC --------------------------------------------------------
    if cv2.waitKey(1) & 0xFF == 27:
        break

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------
cap.release()
cv2.destroyAllWindows()
pygame.mixer.music.stop()
pygame.mixer.quit()