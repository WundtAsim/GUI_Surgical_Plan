import copy

import open3d as o3d
import numpy as np

class Extraction():
    def __init__(self, pcd, debug=False):
        self.pcd = pcd
        self.debug = debug
        self.center = [] # center points of the raw
        self.skeleton = o3d.geometry.PointCloud() # raw of the skeleton

    def _computePlane(self, p1, p2, p3):
        x1, y1, z1 = p1
        x2, y2, z2 = p2
        x3, y3, z3 = p3
        A = y1 * (z2 - z3) + y2 * (z3 - z1) + y3 * (z1 - z2)
        B = z1 * (x2 - x3) + z2 * (x3 - x1) + z3 * (x1 - x2)
        C = x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)
        D = -(x1*(y2*z3 - y3*z2) + x2*(y3*z1 - y1*z3) + x3*(y1*z2 - y2*z1))
        return [A,B,C,D]
    def _distPlane(self, A,B,C,D,x,point):
        # compute the distance between point and plane
        return A*(point[0]-x)+B*point[1]+C*point[2]+D

    def get_slice(self, x, pose_path, len_instrucment):
        points = []
        points.append(np.array([x, 0, 100]))
        points.append(np.array([x, 100, 0]))
        points.append(np.array([x, 100, 100]))
        A, B, C, D = self._computePlane(points[0], points[1], points[2])
        d = 2
        index = []
        for i, point in enumerate(self.pcd.points):
            if self._distPlane(A, B, C, D, d, point) * self._distPlane(A, B, C, D, -1 * d, point) < 0:
                index.append(i)
        slice = self.pcd.select_by_index(index)
        slice.paint_uniform_color([1, 0, 0])
        rest_points = self.pcd.select_by_index(index, invert=True)

        min_z = slice.get_min_bound()[2]
        max_z = slice.get_max_bound()[2]
        gap = 5
        internal_points = []
        slice_np = np.asarray(slice.points)
        for z in np.arange(min_z, max_z.max(), gap):
            slice_y = slice_np[np.where(np.abs(slice_np[:, 2] - z) < 5)]
            min_y = slice_y[:,1].min()
            max_y = slice_y[:,1].max()
            for y in np.arange(min_y, max_y, gap):
                internal_points.append([x, y, z])
        internal_pcd = o3d.geometry.PointCloud()
        internal_pcd.points = o3d.utility.Vector3dVector(internal_points)
        internal_pcd.paint_uniform_color([1,0,0])

        # get robot based coordinate
        # rotate 30 degree about the x axis
        cam2grp = o3d.geometry.get_rotation_matrix_from_zyx(np.array([180 * np.pi / 180, 0, -30 * np.pi / 180]))
        cam2grp = np.vstack((np.hstack((cam2grp, np.array([[20], [120], [50]]))), np.array([0, 0, 0, 1])))
        pose = np.load(pose_path)
        pose[0][:,-1][:3]*=1000
        cam2base = pose[0] @ cam2grp
        plane_base_pcd = copy.deepcopy(internal_pcd)
        plane_base_pcd.transform(cam2base)
        plane_base_coor = np.asarray(plane_base_pcd.points)
        # for 100mm gripper height
        plane_base_coor[:,2]+=len_instrucment

        return slice+rest_points, internal_pcd, plane_base_coor

    def get_slice_center(self, x):
        d = 2
        pcd_np = np.asarray(self.pcd.points)
        index = np.argwhere(np.abs(pcd_np[:, 0] - x) < d)
        print(index.shape)
        slice = self.pcd.select_by_index(index.squeeze())
        if self.debug:
            slice.paint_uniform_color([1, 0, 0])
            rest_points = self.pcd.select_by_index(index, invert=True)
            rest_points.paint_uniform_color([0, 0, 1])
            # o3d.visualization.draw_geometries([slice, rest_points])
        center = slice.get_center()
        if self.debug:
            print(center)
            axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100, origin=center)
            # o3d.visualization.draw_geometries([slice, axis])
        self.center.append(center)

    def show_center(self):
        center_pcd = o3d.geometry.PointCloud()
        center_pcd.points = o3d.utility.Vector3dVector(np.array(self.center))

    def center_fps(self, npoint):
        xyz = np.array(self.center)
        N, D = xyz.shape
        centroids = np.zeros((npoint,))
        distance = np.ones((N,)) * 1e10
        # farthest = np.random.randint(0, N)
        farthest = np.argmax(xyz,0)[0] # find the max x in raw for init of FPS
        for i in range(npoint):
            centroids[i] = farthest
            centroid = xyz[farthest, :]
            dist = np.sum((xyz - centroid) ** 2, -1)
            mask = dist < distance
            distance[mask] = dist[mask]
            farthest = np.argmax(distance, -1)
        self.skeleton.points = o3d.utility.Vector3dVector(xyz[centroids.astype(np.int32)])


def extract_plane(pcd, pose, distance, len_instrucment):
    # use scan points
    extraction = Extraction(pcd)
    min_x = pcd.get_min_bound()[0]
    max_x = pcd.get_max_bound()[0]
    for i in np.linspace(min_x, max_x, 100):
        x = int(i)
        extraction.get_slice_center(x)
    extraction.show_center()
    extraction.center_fps(3)
    x_all = np.asarray(extraction.skeleton.points)[:,0]
    knee_point_x = np.sort(x_all)[1]
    # take a slice 10cm above the knee
    leg_pcd, plane_pcd, coordinate = extraction.get_slice(knee_point_x + distance, pose, len_instrucment)
    return leg_pcd, plane_pcd, coordinate


