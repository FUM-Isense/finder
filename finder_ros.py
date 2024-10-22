from ultralytics import YOLO
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

CLOSE_RANGE_FRAMES = 3
ROTATE_CAMERA = True
DEBUG = True
LONG_RANGE_FRAMES = 3


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



class ImageDepthSubscriber(Node):
    def __init__(self):
        super().__init__('image_depth_subscriber')

        self.bridge = CvBridge()
        self.current_color_frame = None
        self.current_depth_frame = None

        # Create variables to store subscriptions
        # Subscribe to the topics
        self.color_sub = self.create_subscription(Image, '/camera/color/image_raw', self.color_callback, 10)
        self.depth_sub = self.create_subscription(Image, '/camera/depth/image_rect_raw', self.depth_callback, 10)

    def color_callback(self, color_msg):
        """Callback for the color image"""
        self.current_color_frame = self.bridge.imgmsg_to_cv2(color_msg, "bgr8")
        # self.check_if_frames_received()

    def depth_callback(self, depth_msg):
        """Callback for the depth image"""
        depth_image = self.bridge.imgmsg_to_cv2(depth_msg, "32FC1")  # Depth image format
        
        depth_image_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
        depth_image_8bit = np.uint8(depth_image_normalized)
        
        # Apply a colormap (e.g., COLORMAP_JET or COLORMAP_TURBO)
        self.current_depth_frame = cv2.applyColorMap(depth_image_8bit, cv2.COLORMAP_JET)
        
        # self.check_if_frames_received()

    def get_frame(self):
        """
        Subscribes to the color and depth topics, retrieves the frames, and returns them.
        :return: (color_frame, depth_frame) tuple or (None, None) if frames are not yet available.
        """        
        self.current_color_frame = None
        self.current_depth_frame = None
        # Wait for the frames to be received
        while self.current_color_frame is None or self.current_depth_frame is None:
            rclpy.spin_once(self, timeout_sec=0.1)

        return self.current_color_frame, self.current_depth_frame



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

        if cv2.waitKey(0) & 0xFF == ord('q'):

            return True
        
    finally:
        cv2.destroyAllWindows()


def draw_detection_on_image(img, box, cls):
    cv2.rectangle(img, (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                            (int(box.xyxy[0][2]), int(box.xyxy[0][3])), (255, 0, 0), 2)
    cv2.putText(img, cls,
                (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),
                cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)


def side_check(model, node):

    all_side = []
    counter = 0
    while counter < LONG_RANGE_FRAMES:
        img, depth_colormap = node.get_frame()
        detected_image, results = predict_and_detect(model, img)
        checks = []
        
        if DEBUG:
            debug(detected_image, depth_colormap)
        print(f"{len(results) = }")
        result = results[0]

        if len(result.boxes) < 3:
            print("detection less than 3 ignore capture")
            continue
        
        counter += 1

        for box in result.boxes:
            # checks = []
            print(f"{len(result.boxes) = }")

            # box = result.boxes[0]
            cls = result.names[int(box.cls[0])]

            if DEBUG:
                draw_detection_on_image(img, box, cls)

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
    




def finder(node:ImageDepthSubscriber):
    model_close = YOLO("yolo11n-close-range.pt")
    model_long = YOLO("yolo11n-long-range.pt")

    input("start task")
    close_results = []
    counter = 0
    while counter < CLOSE_RANGE_FRAMES:
        color_image, depth_colormap = node.get_frame()
        detected_image, result = predict_and_detect(model_close, color_image)

        if DEBUG and debug(detected_image, depth_colormap):
            pass
        
        if len(result[0].boxes) == 0:
            print("no detection try again")
            continue

        counter += 1

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
        color_image, depth_colormap = node.get_frame()
        cv2.imshow("frame", color_image)
        if cv2.waitKey(1) & 0xFF == ord(" "):
            break
    
    right_detection = side_check(model_long, node)

    print(right_detection)


    while True:
        color_image, depth_colormap = node.get_frame()
        cv2.imshow("frame", color_image)
        if cv2.waitKey(1) & 0xFF == ord(" "):
            break

    input("start left")

    left_detection = side_check(model_long, node)

    print(left_detection)

    goal = None

    for i in range(len(right_detection)):
        if right_detection[i] == detected_object:
            goal = ("R", i)
            break
    
    if goal is None:
        for i in range(len(left_detection)):
            if left_detection[i] == detected_object:
                goal = ("L", i)
                break
    
    if goal is None:
        print("FAILED")
        exit(0)

    print(goal)


    # for result in close_results:
    #     result.


def main(args=None):
    rclpy.init(args=args)
    node = ImageDepthSubscriber()

    try:

        finder(node)

    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

