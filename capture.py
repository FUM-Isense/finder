import cv2
from read_frame import init_camera, get_frames
import numpy as np


if __name__ == "__main__":
    pipeline = init_camera()

    try:
        counter = 1
        while True:
            color_image, depth_colormap, blob = get_frames(pipeline, roate=False)

            cv2.imshow("frame", color_image)

            key = cv2.waitKey(0) & 0xFF

            if key == ord('1'):
                
                filename = f"long/z_frame_{counter:05d}.jpg"
                print(filename)
                cv2.imwrite(filename, color_image)

                counter += 1

            elif key == 27:
                break

    finally:
        pipeline.stop()
