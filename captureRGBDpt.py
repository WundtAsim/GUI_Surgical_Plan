import pyrealsense2 as rs
import numpy as np
import cv2
import open3d as o3d 
import os
import  UR_TCP_RTDE as UR

view_ind = 0
breakLoopFlag = 0
backgroundColorFlag = 1
pose = []
pcds_path = './data/raw_pcds'
def saveCurrentRGBD(vis):
    global view_ind,depth_image,color_image1,pcd, pose
    # if not os.path.exists('./data/color/'):
    #     os.makedirs('./data/color')
    # if not os.path.exists('./data/depth/'):
    #     os.makedirs('./depth')
    # cv2.imwrite('./data/depth/depth_' + str(view_ind) + '.png', depth_image)
    # cv2.imwrite('./data/color/color_' + str(view_ind) + '.png', color_image1)
    if not os.path.exists(pcds_path):
        os.makedirs(pcds_path)
    o3d.io.write_point_cloud(pcds_path+'/pointcloud_' + str(view_ind) + '.ply', pcd)
    print('No.'+str(view_ind)+' shot is saved.' )
    # get robot pose
    r, t = gripper2base_tcp()
    r = cv2.Rodrigues(r)[0]
    t = t.reshape(-1,1)
    pose_one = np.vstack((np.hstack((r, t)), np.array([0, 0, 0, 1])))
    print(pose_one)
    pose.append(pose_one)
    view_ind+=1
    return False

def breakLoop(vis):
    global breakLoopFlag
    breakLoopFlag +=1
    return False

def change_background_color(vis):
    global backgroundColorFlag
    opt = vis.get_render_option()
    if backgroundColorFlag:
        opt.background_color = np.asarray([0, 0, 0])
        backgroundColorFlag = 0
    else:
        opt.background_color = np.asarray([1, 1, 1])
        backgroundColorFlag = 1
    # background_color ~=backgroundColorFlag
    return False

def gripper2base_tcp():
    # for tcp connected
    TCP_socket = UR.connect('192.168.1.24',30003)
    data = TCP_socket.recv(1116)
    position = UR.get_position(data)
    print('position:=',position)
    pos = position[:3]
    rotation = position[3:] # rotation vector
    UR.disconnect(TCP_socket)
    print("TCP disconnected...")
    return rotation, pos

if __name__=="__main__":
    align = rs.align(rs.stream.color)
    config = rs.config()
    width = 1280
    height = 720
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, 30)
    pipeline = rs.pipeline()
    profile = pipeline.start(config)

    sensor = profile.get_device().first_depth_sensor()
    sensor.set_option(rs.option.enable_auto_exposure, True)
    sensor.set_option(rs.option.enable_auto_white_balance, True)

    # get camera intrinsics
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    # print(intr.width, intr.height, intr.fx, intr.fy, intr.ppx, intr.ppy)
    pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(intr.width, intr.height, intr.fx, intr.fy, intr.ppx, intr.ppy)
    # print(type(pinhole_camera_intrinsic))
    
    geometrie_added = False
    vis = o3d.visualization.VisualizerWithKeyCallback()

    vis.create_window("Pointcloud",width,height)
    pointcloud = o3d.geometry.PointCloud()

    vis.register_key_callback(ord(" "), saveCurrentRGBD)
    vis.register_key_callback(ord("Q"), breakLoop)
    vis.register_key_callback(ord("C"), change_background_color)



    try:
        while True:
            # time_start = time.time()
            pointcloud.clear()
            frameset = pipeline.wait_for_frames()
            aligned_frames = align.process(frameset)
            color_frame = aligned_frames.get_color_frame()
            color_image = np.asanyarray(color_frame.get_data())
            depth_frame = aligned_frames.get_depth_frame()

            depth_frame = rs.decimation_filter(1).process(depth_frame)
            depth_frame = rs.disparity_transform(True).process(depth_frame)
            depth_frame = rs.spatial_filter().process(depth_frame)
            depth_frame = rs.temporal_filter().process(depth_frame)
            depth_frame = rs.disparity_transform(False).process(depth_frame)
            # depth_frame = rs.hole_filling_filter().process(depth_frame)
            
            depth_color_frame = rs.colorizer().colorize(depth_frame)
            depth_color_image = np.asanyarray(depth_color_frame.get_data())

            depth_image = np.asanyarray(depth_frame.get_data())
            color_image1 = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)


            depth = o3d.geometry.Image(depth_image)
            color = o3d.geometry.Image(color_image)

            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color, depth, convert_rgb_to_intensity = False)
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, pinhole_camera_intrinsic)

            pointcloud += pcd

            if not geometrie_added:
                vis.add_geometry(pointcloud)

                geometrie_added = True
            view_control = vis.get_view_control()
            view_control.set_front([0, 0, -1])
            view_control.set_up([0, -1, 0])
            view_control.set_lookat([0, 0, 0])
            view_control.set_zoom(0.01)
            vis.update_geometry(pointcloud)
            vis.poll_events()
            vis.update_renderer()
            '''
            cv2.namedWindow('color image', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('color image', cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR))
            cv2.namedWindow('depth image', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('depth image', depth_color_image )

            key = cv2.waitKey(1)

            # print("FPS = {0}".format(int(1/(time_end-time_start))))

            # press ' ' to save current RGBD images and pointcloud.
            if key & 0xFF == ord(' '):
                # if not os.path.exists('./data/color/'):
                #     os.makedirs('./data/color')
                # if not os.path.exists('./data/depth/'):
                #     os.makedirs('./depth')
                # cv2.imwrite('./data/depth/depth_' + str(view_ind) + '.png', depth_image)
                # cv2.imwrite('./data/color/color_' + str(view_ind) + '.png', color_image1)
                if not os.path.exists(pcds_path):
                    os.makedirs(pcds_path)
                r,t = gripper2base_tcp()
                pose.append(np.vstack((np.hstack((r,t)),np.array([0,0,0,1]))))

                o3d.io.write_point_cloud(pcds_path+'/pointcloud_'+str(view_ind)+'.ply', pcd)
                print('No.'+str(view_ind) + ' shot is saved.' )
                view_ind += 1


            # Press esc or 'q' to close the image window
            elif key & 0xFF == ord('q') or key == 27:
                np.save(pcds_path+"/pose", pose)
                cv2.destroyAllWindows()
                vis.destroy_window()

                break
'''
            if breakLoopFlag:
                np.save(pcds_path+"/pose", pose)
                cv2.destroyAllWindows()
                vis.destroy_window()
                break

            
    finally:
        pipeline.stop()


