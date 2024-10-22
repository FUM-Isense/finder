from ultralytics import YOLO
from read_frame import init_camera, get_frames
import cv2
import time
import numpy as np

def predict(chosen_model, img, classes=[], conf=0.5):
    if classes:
        results = chosen_model.predict(img, classes=classes, conf=conf)
    else:
        results = chosen_model.predict(img, conf=conf)

    return results

def predict_and_detect(chosen_model, img, classes=[], conf=0.5, rectangle_thickness=2, text_thickness=1):
    results = predict(chosen_model, img, classes, conf=conf)
    for result in results:
        for box in result.boxes:
            cv2.rectangle(img, (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                          (int(box.xyxy[0][2]), int(box.xyxy[0][3])), (255, 0, 0), rectangle_thickness)
            cv2.putText(img, f"{result.names[int(box.cls[0])]}",
                        (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),
                        cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), text_thickness)
    return img, results


if __name__ == "__main__":
    model = YOLO("yolo11n-close-range.pt")
    # model = YOLO("best-yolov10b-aug.pt")
    pipeline = init_camera()

    try:
        while True:
            color_image, depth_colormap, blob = get_frames(pipeline, roate=True)

            detected_image, result = predict_and_detect(model, color_image)

            depth_colormap_dim = depth_colormap.shape
            color_colormap_dim = detected_image.shape
            # If depth and color resolutions are different, resize color image to match depth image for display
            if depth_colormap_dim != color_colormap_dim:
                detected_image = cv2.resize(detected_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]),
                                                 interpolation=cv2.INTER_AREA)
                images = np.hstack((detected_image, depth_colormap))
            else:
                images = np.hstack((detected_image, depth_colormap))
            # Show images
            cv2.imshow('Yolo in RealSense', images)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        pipeline.stop()

