import pyrealsense2 as rs
import numpy as np
import cv2
import os
import time
import matplotlib.pyplot as plt

from functions.utils import save_depth_to_txt

class State:
    def __init__(self, *args, **kwargs):
        self.WIN_NAME = 'RealSense'
        self.paused = False

state = State()
def start_capture(save_path):
    saved_count = 0

    pipeline = rs.pipeline()
    config = rs.config()
 
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()

    found_rgb = False
    for s in device.sensors:
        if s.get_info(rs.camera_info.name) == 'RGB Camera':
            found_rgb = True
            break
    if not found_rgb:
        print("The demo requires Depth camera with Color sensor")
        exit(0)
 
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
 
    # config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
    # config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
 
    pipeline.start(config)
 
    pc = rs.pointcloud()
    points = rs.points()
 
    align_to = rs.stream.color 
    align = rs.align(align_to)
    plt.ion()
    fig, ax = plt.subplots()

    def get_aligned_images():
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        aligned_color_frame = aligned_frames.get_color_frame()
    
        img_color = np.asanyarray(aligned_color_frame.get_data())
        img_depth = np.asanyarray(aligned_depth_frame.get_data())

        # depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics
        # color_intrin = aligned_color_frame.profile.as_video_stream_profile().intrinsics
    
        depth_mapped_image = cv2.applyColorMap(cv2.convertScaleAbs(img_depth, alpha=0.03), cv2.COLORMAP_JET)
    
        # return color_intrin, depth_intrin, img_color, img_depth, depth_mapped_image, aligned_color_frame, aligned_depth_frame
        return img_color, img_depth, depth_mapped_image, aligned_color_frame, aligned_depth_frame

    def get_3d_camera_coordinates(aligned_color_frame, aligned_depth_frame):
        pc.map_to(aligned_color_frame)
        points = pc.calculate(aligned_depth_frame)
        vtx = np.asanyarray(points.get_vertices())
        vtx = np.reshape(vtx, (480, 640, -1))
        return vtx

    def on_key(event):
        global saved_count, state
        if event.key == 'q' or event.key == 'escape':
            # Exit loop
            state.paused ^= True
            plt.ioff()  # Close interactive mode if needed
            plt.close()
    fig.canvas.mpl_connect('key_press_event', on_key)

    print("Press 'q' to exit the capture.")
    while not state.paused:
        img_color, img_depth, depth_mapped_image, aligned_color_frame, aligned_depth_frame = get_aligned_images()

        roi_x_min = 200
        roi_x_max = 430 
        roi_y_min = 150
        roi_y_max = 400
        ax.imshow(np.hstack((img_color, depth_mapped_image)))
        ax.add_patch(plt.Rectangle((roi_x_min - 3, roi_y_min - 3), roi_x_max - roi_x_min + 6, roi_y_max - roi_y_min + 6,
                                edgecolor='r', facecolor='none'))
        plt.draw()
        plt.pause(0.001)
        ax.clear()

        roi_color = img_color[roi_y_min:roi_y_max, roi_x_min:roi_x_max]
        saved_color_image = img_color
        
        point_cloud = get_3d_camera_coordinates(aligned_color_frame, aligned_depth_frame)
        roi_point_cloud = point_cloud[roi_y_min:roi_y_max, roi_x_min:roi_x_max]

        cenx, ceny = img_color.shape[1] // 2, img_color.shape[0] // 2
        camera_coordinate = point_cloud[ceny, cenx]  # 中心点的3D坐标
        print("Camera coordinate at center:", camera_coordinate)

    cv2.imwrite(os.path.join(save_path, "roi_color.png"), roi_color)
    cv2.imwrite(os.path.join(save_path, "saved_image.png"), saved_color_image)
    pipeline.stop()
    cv2.destroyAllWindows()
    # return point_cloud
    return roi_point_cloud

if __name__ == "__main__":
    point_cloud = start_capture("output")
    print("Point cloud shape:", point_cloud.shape)
    print("Point cloud data:", point_cloud[100, 100, :])