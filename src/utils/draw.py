import cv2

def draw_box(frame, box, label):
    x1, y1, x2, y2 = map(int, box)

    if label == "Low":
        color = (0, 255, 0)       
    elif label == "Medium":
        color = (0, 255, 255)     
    elif label == "High":
        color = (0, 0, 255)       
    else:  
        color = (0, 0, 128)        

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        frame,
        f"REBA {label}",
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2
    )


def draw_keypoints(frame, keypoints):
    for x, y in keypoints:
        if x > 0 and y > 0:
            cv2.circle(frame, (int(x), int(y)), 3, (255, 255, 0), -1)
