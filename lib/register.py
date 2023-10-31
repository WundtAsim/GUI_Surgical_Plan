import os
import open3d as o3d
import copy
import numpy as np

def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh
def load_point_clouds(path, voxel_size=0.0):
    pcd_paths = os.listdir(path)
    pcd_paths.sort()
    pcd_paths = [path+'/' + i for i in pcd_paths]
    pcds = []
    pcds_down = []
    pcds_fpfh = []
    for i in pcd_paths:
        pcd = o3d.io.read_point_cloud(i, 'auto', True, True)
        pcds.append(pcd)
        pcd_down, pcd_fpfh = preprocess_point_cloud(pcd, voxel_size)
        pcds_down.append(pcd_down)
        pcds_fpfh.append(pcd_fpfh)
    return pcds, pcds_down, pcds_fpfh

def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result

# Local Refinement
def refine_registration(source, target, result_coarse, voxel_size):
    distance_threshold = voxel_size * 0.4
    print(":: Point-to-plane ICP registration is applied on original point")
    print("   clouds to refine the alignment. This time we use a strict")
    print("   distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, result_coarse,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return result

def execute_fast_global_registration(source_down, target_down, source_fpfh,
                                     target_fpfh, voxel_size):
    distance_threshold = voxel_size * 0.5
    print(":: Apply fast global registration with distance threshold %.3f" \
            % distance_threshold)
    result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold))
    return result

def pairwise_registration(source, target, source_fpfh, target_fpfh, voxel_size):
    print("Apply point-to-plane ICP")
    icp_coarse = execute_fast_global_registration(source, target,
                                            source_fpfh, target_fpfh,
                                            voxel_size)
    icp_fine = refine_registration(source, target, icp_coarse.transformation, voxel_size)
    transformation_icp = icp_fine.transformation
    return transformation_icp

def full_registration(pcds, pcds_fpfh, voxel_size):
    pcd_combined = o3d.geometry.PointCloud()

    for i in range(len(pcds)):
        if i == 0:
            pcd_combined += pcds[0]
        else:
            pcd, pcd_fpfh = preprocess_point_cloud(pcd_combined, voxel_size)
            # transform from pcds[i] to the current raw
            tran = pairwise_registration(pcds[i], pcd, pcds_fpfh[i], pcd_fpfh, voxel_size)
            pcds[i].transform(tran)
            pcd_combined += pcds[i]
            # o3d.visualization.draw_geometries([pcd_combined], window_name='combined')
    return pcd_combined

def full_registration_pose(pcds, pcds_fpfh, voxel_size, pose_file):
    pcd_combined = o3d.geometry.PointCloud()
    pose = np.load(pose_file)
    # rotate 30 degree about the x axis
    cam2grp = o3d.geometry.get_rotation_matrix_from_zyx(np.array([180*np.pi/180, 0, -30*np.pi/180]))
    cam2grp = np.vstack((np.hstack((cam2grp, np.array([[20],[120],[50]]))), np.array([0,0,0,1])))
    print(pose)
    for i in range(len(pcds)):
        pose[i][:,-1][:3]*=1000
        print(pose)
        if i == 0:
            pcd_combined += pcds[0]
        else:
            pcd, pcd_fpfh = preprocess_point_cloud(pcd_combined, voxel_size)
            # transform from pcds[i] to the current raw

            transformation_coarse = np.linalg.inv(cam2grp) @ np.linalg.inv(pose[0]) @ pose[i] @ cam2grp
            icp_fine = refine_registration(pcds[i], pcd, transformation_coarse, voxel_size)
            transformation_icp = icp_fine.transformation

            pcds[i].transform(transformation_icp)
            pcd_combined += pcds[i]
    return pcd_combined


def register(pose_file, pcds_path):
    voxel_size = 5
    pcds, pcds_down, pcds_fpfh = load_point_clouds(pcds_path, voxel_size)
    print("Full registration ...")
    pcd_combined = full_registration_pose(pcds_down, pcds_fpfh, voxel_size, pose_file)
    return pcd_combined






