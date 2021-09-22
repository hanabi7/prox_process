# This is a sample Python script.
import numpy as np
import os
import random
import os.path as osp
import glob
import cv2
import json


def unproject_depth_image(self, depth_image, cam):
    us = np.arange(depth_image.size) % depth_image.shape[1]
    vs = np.arange(depth_image.size) // depth_image.shape[1]
    ds = depth_image.ravel()
    uvd = np.array(np.vstack((us.ravel(), vs.ravel(), ds.ravel())).T)
    # unproject
    xy_undistorted_camspace = cv2.undistortPoints(np.asarray(uvd[:, :2].reshape((1, -1, 2)).copy()),
                                                  np.asarray(cam['camera_mtx']), np.asarray(cam['k']))
    xyz_camera_space = np.hstack((xy_undistorted_camspace.squeeze(), self.col(uvd[:, 2])))
    xyz_camera_space[:, :2] *= self.col(xyz_camera_space[:, 2])  # scale x,y by z
    other_answer = xyz_camera_space - self.row(np.asarray(cam['view_mtx'])[:, 3])  # translate
    xyz = other_answer.dot(np.asarray(cam['view_mtx'])[:, :3])  # rotate


base_dir = 'E:\Working\Datasets\Prox'
video_path = osp.join(base_dir, 'videos')
extrinsic_path = osp.join(base_dir, 'cam2world', 'cam2world')
calibration_path = osp.join(base_dir, 'calibration', 'calibration')
video_ext = '.mp4'
extrinsic_ext = '.json'
demo_videos_list = glob.glob(video_path + '\*' + video_ext)
demo_video = demo_videos_list[0]
demo_name = demo_video.split('\\')[-1]
demo_name = demo_name.split('.')[0]
demo_name = demo_name.split('_')[0]
print(demo_name)
demo_extrinsic_path = osp.join(extrinsic_path, demo_name + extrinsic_ext)
with open(demo_extrinsic_path) as f:
    extrinsic_file = json.load(f)
extrinsic_matrix = np.zeros((4, 4))
for i in range(4):
    for j in range(4):
        extrinsic_matrix[j, i] = extrinsic_file[i][j]
print(extrinsic_matrix)
with open(osp.join(calibration_path, 'Color.json')) as f:
    color_file = json.load(f)
with open(osp.join(calibration_path, 'IR.json')) as f:
    ir_file = json.load(f)
print('color_file:', color_file)
print('ir_file:', ir_file)
cap = cv2.VideoCapture(demo_video)
cut = 1920
print('Processing demo video:', demo_video)
# [1080, 5760, 3]
ret, frame = cap.read()
if frame is not None:
    frame = frame[:, cut:2*cut, :]
    sample_array = (frame)
    cv2.imshow('image', frame)
    k = cv2.waitKey(1)
cap.release()
cv2.destroyAllWindows()
