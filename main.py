import logging as log
import threading
import argparse
import pyrealsense2 as rs
import  UR_TCP_RTDE as UR
import numpy as np
import socket
import open3d.visualization.gui as gui
from lib.camera import PipelineModel
from lib.gui import AppWindow


class PipelineController:
    """Entry point for the app. Controls the PipelineModel object for IO and
    processing  and the PipelineView object for display and UI. All methods
    operate on the main thread.
    """

    def __init__(self, camera_config_file=None, device=None):
        """Initialize.

        Args:
            camera_config_file (str): Camera configuration json file.
            device (str): Compute device (e.g.: 'cpu:0' or 'cuda:0').
        """
        self.pose = []
        ctx = rs.context()
        cameras = ctx.query_devices()
        if len(cameras) == 1:
            self.pipeline_model = PipelineModel(self.update_view,
                                            camera_config_file,
                                            device)
            self.pipeline_view = AppWindow(
                1.25 * self.pipeline_model.vfov,
                self.pipeline_model.max_points,
                on_window_close=self.on_window_close,
                on_toggle_camera=self.on_toggle_camera,
                on_save_pcd=self.on_save_pcd,
                on_save_rgbd=self.on_save_rgbd)
            threading.Thread(name='PipelineModel',
                             target=self.pipeline_model.run).start()
        else:
            self.pipeline_view = AppWindow()


        self.pipeline_view.run()

    def update_view(self, frame_elements):
        """Updates view with new data. May be called from any thread.

        Args:
            frame_elements (dict): Display elements (point cloud and images)
                from the new frame to be shown.
        """
        gui.Application.instance.post_to_main_thread(
            self.pipeline_view.window,
            lambda: self.pipeline_view.update(frame_elements))

    def on_toggle_camera(self, is_enabled):
        """Callback to toggle capture."""
        self.pipeline_model.flag_capture = is_enabled
        if is_enabled:
            with self.pipeline_model.cv_capture:
                self.pipeline_model.cv_capture.notify()
        self.pipeline_view._apply_settings()

    def on_window_close(self):
        """Callback when the user closes the application window."""
        self.pipeline_model.flag_exit = True
        with self.pipeline_model.cv_capture:
            self.pipeline_model.cv_capture.notify_all()
        return True  # OK to close window

    def on_save_pcd(self):
        """Callback to save current point cloud."""
        self.pipeline_model.flag_save_pcd = True
        r, t = self.gripper2base_tcp()
        if r and t:
            self.pose.append(np.vstack((np.hstack((r, t)), np.array([0, 0, 0, 1]))))
            np.save('data/pose', self.pose)
        else:
            print("[ROBOT missing]: without robot pose")

    def on_save_rgbd(self):
        """Callback to save current RGBD image pair."""
        self.pipeline_model.flag_save_rgbd = True

    def gripper2base_tcp(self):
        # for tcp connected
        def is_connected(host, port):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(0.5)
            result = sock.connect_ex(('192.168.1.24', 30003)) == 1
            sock.close()
            return result
        if is_connected('192.168.1.24', 30003):
            TCP_socket = UR.connect('192.168.1.24', 30003)
            data = TCP_socket.recv(1116)
            position = UR.get_position(data)
            print('position:=', position)
            pos = position[:3]
            rotation = position[3:]  # rotation vector
            UR.disconnect(TCP_socket)
            print("TCP disconnected...")
            return rotation, pos
        else:
            return None, None


if __name__ == "__main__":

    log.basicConfig(level=log.INFO)
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--camera-config',
                        help='RGBD camera configuration JSON file')
    parser.add_argument('--device',
                        help='Device to run computations. e.g. cpu:0 or cuda:0 '
                        'Default is CUDA GPU if available, else CPU.')

    args = parser.parse_args()
    PipelineController(args.camera_config, args.device)