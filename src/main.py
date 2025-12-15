import cv2
from ultralytics import YOLO

from ergonomics.reba import reba_lite_score, classify_reba
from pose.ergonomics import is_valid_pose
from utils.draw import draw_box, draw_keypoints

VIDEO_PATH = "data/videos/test.mp4"

def main():
    model = YOLO("yolov8s-pose.pt")
    cap = cv2.VideoCapture(VIDEO_PATH)

    cv2.namedWindow("Pose Ergonomics Application", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Pose Ergonomics Application", 1280, 720)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=0.2, verbose=False)

        for r in results:
            if r.boxes is None or r.keypoints is None:
                continue

            boxes = r.boxes.xyxy.cpu().numpy()
            keypoints = r.keypoints.xy.cpu().numpy()

            for box, kp in zip(boxes, keypoints):
                points = kp.tolist()

                if not is_valid_pose(points):
                    continue

                score = reba_lite_score(points)
                label = classify_reba(score)

                draw_box(frame, box, label)
                draw_keypoints(frame, points)

        cv2.imshow("Pose Ergonomics Application", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
