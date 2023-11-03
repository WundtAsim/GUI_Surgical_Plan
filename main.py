import logging as log
import threading
import argparse
import pyrealsense2 as rs
import open3d.visualization.gui as gui

from lib.camera import PipelineModel
from lib.gui import AppWindow


class PipelineController:
    """Entry point for the app. Controls the PipelineModel object for IO and
    processing  and the PipelineView object for display and UI. All methods
    operate on the main thread.
    """

    def __init__(self, camera_config_file='./lib/condig.json', device=None):
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

    def on_save_rgbd(self):
        """Callback to save current RGBD image pair."""
        self.pipeline_model.flag_save_rgbd = True


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