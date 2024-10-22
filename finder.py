from yolov11 import predict_and_detect
from read_frame import init_camera, get_frames
from ultralytics import YOLO
import cv2
import numpy as np
import time

CLOSE_RANGE_FRAMES = 3
ROTATE_CAMERA = True
DEBUG = True
LONG_RANGE_FRAMES = 3


def debug(detected_image, depth_colormap):
    try:
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

        if cv2.waitKey(0) & 0xFF == ord(' '):

            return True
        
    finally:
        cv2.destroyAllWindows()



def side_check(model, pipeline):

    all_side = []

    for _ in range(LONG_RANGE_FRAMES):
        img, depth_colormap, blob = get_frames(pipeline, roate=ROTATE_CAMERA)
        detected_image, results = predict_and_detect(model, img)
        checks = []

        if DEBUG:
            debug(detected_image, depth_colormap)
        print(f"{len(results) = }")
        result = results[0]
        for box in result.boxes:
            # checks = []
            print(f"{len(result.boxes) = }")

            # box = result.boxes[0]
            cls = result.names[int(box.cls[0])]

            if DEBUG:
                cv2.rectangle(img, (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                            (int(box.xyxy[0][2]), int(box.xyxy[0][3])), (255, 0, 0), 2)
                cv2.putText(img, f"{result.names[int(box.cls[0])]}",
                            (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),
                            cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)

            confidence = float(box.conf[0])
            
            detection = (cls, confidence, int(box.xyxy[0][1]))
            
            all_side.append(detection)

        

        if DEBUG and debug(img, depth_colormap):
            pass


    detected_poses = {}
    print(all_side)

    for detection in all_side:
        if detection[0] in detected_poses.keys() and detection[1] > detected_poses[detection[0]][1]:
            detected_poses[detection[0]] = detection
        
        elif detection[0] not in detected_poses.keys():
            detected_poses[detection[0]] = detection
    

    final =  list(detected_poses.values())
    final.sort(key=lambda x: -x[1])
    final = final[:3]
    final.sort(key=lambda x:x[2])

    return final
    




def finder():
    model_close = YOLO("yolo11n-close-range.pt")
    model_long = YOLO("yolo11n-long-range.pt")

    pipeline = init_camera()

    input("start task")
    close_results = []

    for i in range(CLOSE_RANGE_FRAMES):
        color_image, depth_colormap, blob = get_frames(pipeline, roate=ROTATE_CAMERA)
        detected_image, result = predict_and_detect(model_close, color_image)

        if DEBUG and debug(detected_image, depth_colormap):
            pass

        detected_object = result[0].names[int(result[0].boxes[0].cls[0])]
        conf = float(result[0].boxes[0].conf[0])
        
        close_results.append((detected_object, conf))

        print(f"{conf = }")
        print("==================================")
        

    
    close_results.sort(key=lambda x:-x[1])
    
    
    detected_object = close_results[0][0]
    print(detected_object)
    
    cv2.destroyAllWindows()

    input("start frame")

    # input("start right")
    while True:
        color_image, depth_colormap, blob = get_frames(pipeline, roate=ROTATE_CAMERA)
        cv2.imshow("frame", color_image)
        if cv2.waitKey(1) & 0xFF == ord(" "):
            break
    
    right_detection = side_check(model_long, pipeline)

    print(right_detection)


    while True:
        color_image, depth_colormap, blob = get_frames(pipeline, roate=ROTATE_CAMERA)
        cv2.imshow("frame", color_image)
        if cv2.waitKey(1) & 0xFF == ord(" "):
            break

    input("start left")

    left_detection = side_check(model_long, pipeline)

    print(left_detection)

    goal = None

    for i in range(right_detection):
        if right_detection[i] == detected_object:
            goal = ("R", i)
            break
    
    if goal is None:
        for i in range(left_detection):
            if left_detection[i] == detected_object:
                goal = ("L", i)
                break
    
    if goal is None:
        print("FAILED")
        exit(0)

    print(goal)


    # for result in close_results:
    #     result.


if __name__ == "__main__":
    finder()