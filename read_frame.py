import pyrealsense2 as rs
import sys
import numpy as np
import cv2

confThreshold = 0.5
nmsThreshold = 0.4
inpWidth = 416
inpHeight = 416


def init_camera():
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()

    # Get device product line for setting a supporting resolution
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))
    depth_sensor = pipeline_profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    found_rgb = False

    for s in device.sensors:
        if s.get_info(rs.camera_info.name) == 'RGB Camera':
            found_rgb = True
            break
    if not found_rgb:
        print("The demo requires Depth camera with Color sensor")
        sys.exit()

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    if device_product_line == 'L500':
        config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
    else:
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    # Start streaming
    pipeline.start(config)

    return pipeline


def get_frames(pipeline, roate=False):
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    
    if not depth_frame or not color_frame:
        return None, None
    # Convert images to numpy arrays
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    if roate:
        color_image = cv2.rotate(color_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        depth_image = cv2.rotate(depth_image, cv2.ROTATE_90_COUNTERCLOCKWISE)

    blob = cv2.dnn.blobFromImage(color_image, 1/255, (inpWidth, inpHeight), [0,0,0],1,crop=False)
    # net.setInput(blob)
    # outs = net.forward(getOutputsNames(net))
    # print(len(color_image))
    # outs = model(color_image)

    # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
    return color_image, depth_colormap, blob

