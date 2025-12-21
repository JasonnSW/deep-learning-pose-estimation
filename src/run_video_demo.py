import cv2
import numpy as np
from ultralytics import YOLO

from ergonomics.reba import compute_angles, compute_reba, classify_reba
from utils.draw import draw_keypoints, draw_box


def main():
    model = YOLO("yolov8s-pose.pt")
    cap = cv2.VideoCapture("data/videos/test.mp4")  # Change to: test.mp4 / test2.mp4 / test3.mp4 / test4.mp4

    cv2.namedWindow("Ergonomic Risk Analysis", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Ergonomic Risk Analysis", 1280, 720)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=0.3)[0]

        if results.keypoints is not None:
            keypoints_all = results.keypoints.xy.cpu().numpy()
            boxes = results.boxes.xyxy.cpu().numpy()

            for kp, box in zip(keypoints_all, boxes):
                angles = compute_angles(kp)
                reba_score = compute_reba(angles)
                risk = classify_reba(reba_score)

                label = f"REBA {reba_score} ({risk})"
                draw_keypoints(frame, kp)
                draw_box(frame, box, label)

        cv2.imshow("Ergonomic Risk Analysis", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
