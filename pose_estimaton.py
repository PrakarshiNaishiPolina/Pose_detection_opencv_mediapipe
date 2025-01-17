import cv2 as cv
import mediapipe as mp

# initialize pose module

pose=mp.solutions.pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
    )

# Initialize drawing utilities for drawing landmarks

mp_drawing=mp.solutions.drawing_utils

cap=cv.VideoCapture(0)

while cap.isOpened():
    ret,frame=cap.read()
    if not ret:
        break
    # flip the frame horizontally

    frame=cv.flip(frame,1)

    # convert the frame to RGB

    rgb_frame=cv.cvtColor(frame,cv.COLOR_BGR2RGB)

    # process the frame with Pose module

    results=pose.process(rgb_frame)

    # if pose landmarks are found, draw the landmarks on the frame

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame,results.pose_landmarks,mp.solutions.pose.POSE_CONNECTIONS)

    cv.imshow("Pose Detection",frame)

    # exit on pressing 'd'

    if cv.waitKey(1) & 0xFF == ord('d'):
        break

# release video capture and close any OpenCV windows

cap.release()
cv.destroyAllWindows()