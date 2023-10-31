import open3d as o3d
import numpy as np
from .settings import Settings


# 定义直通滤波函数
def pass_through(cloud, limit_min=0., limit_max=10., filter_value_name="z"):
    points = np.asarray(cloud.points)
    if filter_value_name == "x":
        ind = np.where((points[:, 0] >= limit_min) & (points[:, 0] <= limit_max))[0]
        x_cloud = cloud.select_by_index(ind)
        return x_cloud
    elif filter_value_name == "y":
        ind = np.where((points[:, 1] >= limit_min) & (points[:, 1] <= limit_max))[0]
        y_cloud = cloud.select_by_index(ind)
        return y_cloud
    elif filter_value_name == "z":
        ind = np.where((points[:, 2] >= limit_min) & (points[:, 2] <= limit_max))[0]
        z_cloud = cloud.select_by_index(ind)
        return z_cloud

def show(pcd, name):
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(200)
    o3d.visualization.draw_geometries([pcd, axis],
                                      window_name=name,
                                      width=1280, height=720,
                                      point_show_normal=False,
                                      lookat=[0, 0, 0],
                                      up=[0, -1, 0],
                                      front=[0, 0, -1],
                                      zoom=0.5)

def segment_pcd(pcd, settings:Settings):
    # 直通
    if settings.pass_through:
        pcd = pass_through(pcd,
                           -1*settings.pass_x,
                           1*settings.pass_x,'x')
        pcd = pass_through(pcd,
                           -1*settings.pass_y,
                           1*settings.pass_y,'y')
        pcd = pass_through(pcd,
                           0,
                           settings.pass_z,"z")

    # 去噪
    if settings.noise_reduct:
        # static: nb_neighbors:最近k个点    std_ratio:基于标准差的阈值，越小滤除点越多
        cl, ind = pcd.remove_statistical_outlier(
            nb_neighbors=settings.noise_nb_neighbors,
            std_ratio=settings.noise_std_radio)
        pcd = pcd.select_by_index(ind)
        # radius: nb_points:基于球体内包含点数量的阈值  radius:半径
        # cl, ind = filtered_cloud.remove_radius_outlier(nb_points=100, radius=0.04*unit)
        # filtered_cloud = filtered_cloud.select_by_index(ind)

    # 降采样
    if settings.down_sample:
        # voxel: center of mass
        pcd = pcd.voxel_down_sample(voxel_size=settings.down_sample_size)
        # uniform down sample
        pcd = pcd.uniform_down_sample(every_k_points=1)

    # RANSAC: random sample consensus
    if settings.rm_plane:
        cl, ind = pcd.segment_plane(
            settings.rm_plane_d,
            settings.rm_plane_min_points,
            10000) # 距离阈值，模型点个数， 随机次数
        pcd = pcd.select_by_index(ind, invert=True)

    # DBSCAN: Denisity-based spatial clustering
    if settings.dbscan_seg:
        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
            labels = np.array(pcd.cluster_dbscan(
                eps=settings.dbscan_seg_eps,
                min_points=settings.dbscan_seg_min_points))
        min_label = labels.min()
        max_label = labels.max()
        leg_label = []
        for i in range(min_label, max_label+1):
            label_index = np.array(np.where(labels == i))[0]
            leg_label = label_index if len(label_index)>len(leg_label) else leg_label
        pcd = pcd.select_by_index(leg_label)

    # random down sample
    if settings.down_sample:
        pcd = pcd.random_down_sample(settings.down_sample_points / len(pcd.points))

    print(pcd)
    return pcd



