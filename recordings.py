import numpy as np
import sys
import matplotlib.pyplot as plt
import os
import glob
import json
import os.path as osp
import cv2
import random
from plyfile import PlyData, PlyElement


def col(tensor):
    return tensor.reshape(1, -1)


def row(tensor):
    return tensor.reshape(-1, 1)


def unproject_depth_image(depth_image, cam):
    us = np.arange(depth_image.size) % depth_image.shape[1]
    vs = np.arange(depth_image.size) // depth_image.shape[1]
    ds = depth_image.ravel()
    uvd = np.array(np.vstack((us.ravel(), vs.ravel(), ds.ravel())).T)
    # unproject
    xy_undistorted_camspace = cv2.undistortPoints(np.asarray(uvd[:, :2].reshape((1, -1, 2)).copy()),
                                                  np.asarray(cam['camera_mtx']), np.asarray(cam['k']))
    xyz_camera_space = np.hstack((xy_undistorted_camspace.squeeze(), row(uvd[:, 2])))
    xyz_camera_space[:, :2] *= row(xyz_camera_space[:, 2])  # scale x,y by z
    other_answer = xyz_camera_space - col(np.asarray(cam['view_mtx'])[:, 3])  # translate
    xyz = other_answer.dot(np.asarray(cam['view_mtx'])[:, :3])  # rotate
    return xyz.reshape(-1, 3)


def write_point_cloud(vertices, filename, colors=None, labels=None):
    assert (colors is None or labels is None)
    verts_num = vertices.shape[0]
    verts = [(vertices[i, 0], vertices[i, 1], vertices[i, 2]) for i in range(verts_num)]
    verts = np.array(verts, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    el = PlyElement.describe(verts, 'vertex')
    PlyData([el], text=False).write(filename)

base_dir = 'F:\\recordings\BasementSittingBooth_00142_01'
calibration_file = osp.join(base_dir, 'Calibration.txt')
skeleton_dir = osp.join(base_dir, 'Skeleton')
num_joints = 25
depth_dir = osp.join(base_dir, 'Depth')
depth_image_list = glob.glob(depth_dir + '\*' + '.png')
cam_to_world_dir = 'F:\cam2world\cam2world\\' + 'BasementSittingBooth.json'
print('total frames:', len(depth_image_list))
with open(osp.join(skeleton_dir, 's001_frame_00001__00.00.00.029.json')) as f:
    skeleton_file = json.load(f)
joints = skeleton_file['Bodies'][0]['Joints']
keys_list = list(joints.keys())
# dict_keys(['SpineBase', 'SpineMid', 'Neck', 'Head', 'ShoulderLeft', 'ElbowLeft', 'WristLeft', 'HandLeft',
# 'ShoulderRight', 'ElbowRight', 'WristRight', 'HandRight', 'HipLeft', 'KneeLeft', 'AnkleLeft', 'FootLeft',
# 'HipRight', 'KneeRight', 'AnkleRight', 'FootRight', 'SpineShoulder', 'HandTipLeft', 'ThumbLeft', 'HandTipRight',
# 'ThumbRight'])
joints_position = np.zeros((num_joints, 3))
for index in range(num_joints):
    joint_name = keys_list[index]
    # print(joint_name)
    single_joint = joints[joint_name]
    # dict_keys(['Position', 'State', 'Rotation'])
    joints_position[index, :] = single_joint['Position']
# print(joints_position)
single_depth_image = depth_image_list[0]

with open(cam_to_world_dir) as f:
    cam_to_world_list = json.load(f)
    cam_to_world = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            cam_to_world[j, i] = cam_to_world_list[i][j]
    print(cam_to_world)

calib_dir = 'F:\calibration\calibration'
with open(osp.join(calib_dir, 'IR.json'), 'r') as f:
    depth_cam = json.load(f)
file_name = 'F:\\recordings\point_cloud.ply'
depth_image = cv2.imread(single_depth_image, -1).astype(float)
depth_image /= 8.0
depth_image /= 1000.0
depth_image = cv2.flip(depth_image, 1)
point_cloud = unproject_depth_image(depth_image, depth_cam)
print(point_cloud.max())
write_point_cloud(point_cloud, file_name)
skeleton_file_name = 'F:\\recordings\skeleton.ply'
print(joints_position)
write_point_cloud(joints_position, skeleton_file_name)
