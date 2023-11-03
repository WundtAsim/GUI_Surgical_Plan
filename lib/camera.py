import json
import time
import logging as log
import threading
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import open3d as o3d
import  UR_TCP_RTDE as UR
import socket

# Camera and processing
class PipelineModel:
    """Controls IO (camera, video file, recording, saving frames). Methods run
    in worker threads."""

    def __init__(self,
                 update_view,
                 camera_config_file=None,
                 device=None):
        """Initialize.

        Args:
            update_view (callback): Callback to update display elements for a
                frame.
            camera_config_file (str): Camera configuration json file.
            rgbd_video (str): RS bag file containing the RGBD video. If this is
                provided, connected cameras are ignored.
            device (str): Compute device (e.g.: 'cpu:0' or 'cuda:0').
        """
        # new: robot pose
        self.pose = []
        self.update_view = update_view
        if device:
            self.device = device.lower()
        else:
            self.device = 'cuda:0' if o3d.core.cuda.is_available() else 'cpu:0'
        self.o3d_device = o3d.core.Device(self.device)

        self.video = None
        self.camera = None
        self.flag_capture = False
        self.cv_capture = threading.Condition()  # condition variable
        self.recording = False  # Are we currently recording
        self.flag_record = False  # Request to start/stop recording

        # RGBD camera
        now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        filename = f"{now}.bag"
        self.camera = o3d.t.io.RealSenseSensor()
        if camera_config_file:
            with open(camera_config_file) as ccf:
                self.camera.init_sensor(o3d.t.io.RealSenseSensorConfig(
                    json.load(ccf)), filename=filename)
        else:
            self.camera.init_sensor(filename='')
        self.camera.start_capture(start_record=False)
        self.rgbd_metadata = self.camera.get_metadata()
        self.status_message = f"Camera {self.rgbd_metadata.serial_number} opened."

        log.info(self.rgbd_metadata)

        # RGBD -> PCD
        self.extrinsics = o3d.core.Tensor.eye(4,
                                              dtype=o3d.core.Dtype.Float32,
                                              device=self.o3d_device)
        self.intrinsic_matrix = o3d.core.Tensor(
            self.rgbd_metadata.intrinsics.intrinsic_matrix,
            dtype=o3d.core.Dtype.Float32,
            device=self.o3d_device)
        self.depth_max = 3.0  # m
        self.pcd_stride = 2  # downsample point cloud, may increase frame rate
        self.flag_normals = False
        self.flag_save_rgbd = False
        self.flag_save_pcd = False

        self.pcd_frame = None
        self.rgbd_frame = None
        self.executor = ThreadPoolExecutor(max_workers=5,
                                           thread_name_prefix='Capture-Save')
        self.flag_exit = False

    @property
    def max_points(self):
        """Max points in one frame for the camera or RGBD video resolution."""
        return self.rgbd_metadata.width * self.rgbd_metadata.height

    @property
    def vfov(self):
        """Camera or RGBD video vertical field of view."""
        return np.rad2deg(2 * np.arctan(self.intrinsic_matrix[1, 2].item() /
                                        self.intrinsic_matrix[1, 1].item()))

    def run(self):
        """Run pipeline."""
        n_pts = 0
        frame_id = 0
        t1 = time.perf_counter()
        if self.video:
            self.rgbd_frame = self.video.next_frame()
        else:
            self.rgbd_frame = self.camera.capture_frame(
                wait=True, align_depth_to_color=True)

        pcd_errors = 0
        while (not self.flag_exit and
               (self.video is None or  # Camera
                (self.video and not self.video.is_eof()))):  # Video
            if self.video:
                future_rgbd_frame = self.executor.submit(self.video.next_frame)
            else:
                future_rgbd_frame = self.executor.submit(
                    self.camera.capture_frame,
                    wait=True,
                    align_depth_to_color=True)

            if self.flag_save_pcd:
                self.save_pcd()
                self.flag_save_pcd = False
            try:
                self.rgbd_frame = self.rgbd_frame.to(self.o3d_device)
                self.pcd_frame = o3d.t.geometry.PointCloud.create_from_rgbd_image(
                    self.rgbd_frame, self.intrinsic_matrix, self.extrinsics,
                    self.rgbd_metadata.depth_scale, self.depth_max,
                    self.pcd_stride, self.flag_normals)
                depth_in_color = self.rgbd_frame.depth.colorize_depth(
                    self.rgbd_metadata.depth_scale, 0, self.depth_max)
            except RuntimeError:
                pcd_errors += 1

            if self.pcd_frame.is_empty():
                log.warning(f"No valid depth data in frame {frame_id})")
                continue

            n_pts += self.pcd_frame.point.positions.shape[0]
            if frame_id % 60 == 0 and frame_id > 0:
                t0, t1 = t1, time.perf_counter()
                log.debug(f"\nframe_id = {frame_id}, \t {(t1-t0)*1000./60:0.2f}"
                          f"ms/frame \t {(t1-t0)*1e9/n_pts} ms/Mp\t")
                n_pts = 0
            frame_elements = {
                'color': self.rgbd_frame.color.cpu(),
                'depth': depth_in_color.cpu(),
                'raw': self.pcd_frame.cpu(),
                'status_message': self.status_message
            }
            self.update_view(frame_elements)

            if self.flag_save_rgbd:
                self.save_rgbd()
                self.flag_save_rgbd = False
            self.rgbd_frame = future_rgbd_frame.result()
            with self.cv_capture:  # Wait for capture to be enabled
                self.cv_capture.wait_for(
                    predicate=lambda: self.flag_capture or self.flag_exit)

            frame_id += 1

        if self.camera:
            self.camera.stop_capture()
        else:
            self.video.close()
        self.executor.shutdown()
        log.debug(f"create_from_depth_image() errors = {pcd_errors}")

    def save_pcd(self):
        """Save current point cloud."""
        now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        filename = f"data/raw_pcds/{self.rgbd_metadata.serial_number}_pcd_{now}.ply"
        # Convert colors to uint8 for compatibility
        self.pcd_frame.point.colors = (self.pcd_frame.point.colors * 255).to(
            o3d.core.Dtype.UInt8)
        self.executor.submit(o3d.t.io.write_point_cloud,
                             filename,
                             self.pcd_frame,
                             write_ascii=False,
                             compressed=True,
                             print_progress=False)
        self.status_message = f"Saving point cloud to {filename}."

        # robot pose:
        r, t = self.executor.submit(self.gripper2base_tcp).result()
        if r is not None and t is not None:
            r = o3d.geometry.get_rotation_matrix_from_axis_angle(r)
            t = t.reshape(3, 1)
            self.pose.append(np.vstack((np.hstack((r, t)), np.array([0, 0, 0, 1]))))
            np.save('data/pose', self.pose)
        else:
            print("[ROBOT missing]: without robot pose")

    def save_rgbd(self):
        """Save current RGBD image pair."""
        now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        filename = f"{self.rgbd_metadata.serial_number}_color_{now}.jpg"
        self.executor.submit(o3d.t.io.write_image, filename,
                             self.rgbd_frame.color)
        filename = f"{self.rgbd_metadata.serial_number}_depth_{now}.png"
        self.executor.submit(o3d.t.io.write_image, filename,
                             self.rgbd_frame.depth)
        self.status_message = (
            f"Saving RGBD images to {filename[:-3]}.{{jpg,png}}.")

    def gripper2base_tcp(self):
        # for tcp connected
        def is_connected(host, port):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)
            result = sock.connect_ex(('192.168.1.24', 30003)) == 0
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
