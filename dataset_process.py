import numpy as np
import cv2
import json
import argparse
import os
import os.path as osp
import glob
import pickle
import random
from plyfile import PlyData, PlyElement


def col(tensor):
    return tensor.reshape(1, -1)


def row(tensor):
    return tensor.reshape(-1, 1)


def unproject_depth_image(depth_image, depth_mask, cam, sample_number=10000):
    total_number = depth_image.shape[0] * depth_image.shape[1]
    sample_list = np.array(random.sample(range(total_number), sample_number))
    us = np.arange(depth_image.size) % depth_image.shape[1]
    # [424, 512]
    vs = np.arange(depth_image.size) // depth_image.shape[1]
    ds = depth_image.ravel()
    ds_mask = depth_mask.ravel()
    uvd = np.array(np.vstack((us.ravel(), vs.ravel(), ds.ravel())).T)
    # unproject
    xy_undistorted_camspace = cv2.undistortPoints(np.asarray(uvd[:, :2].reshape((1, -1, 2)).copy()),
                                                  np.asarray(cam['camera_mtx']), np.asarray(cam['k']))
    xyz_camera_space = np.hstack((xy_undistorted_camspace.squeeze(), row(uvd[:, 2])))
    xyz_camera_space[:, :2] *= row(xyz_camera_space[:, 2])  # scale x,y by z
    other_answer = xyz_camera_space - col(np.asarray(cam['view_mtx'])[:, 3])  # translate
    xyz = other_answer.dot(np.asarray(cam['view_mtx'])[:, :3])  # rotate
    xyz = xyz.reshape(-1, 3)
    xyz = xyz[sample_list, :]
    ds_mask = ds_mask[sample_list]
    return xyz, ds_mask


def write_point_cloud(vertices, filename, colors=None, labels=None):
    assert (colors is None or labels is None)
    verts_num = vertices.shape[0]
    verts = [(vertices[i, 0], vertices[i, 1], vertices[i, 2]) for i in range(verts_num)]
    verts = np.array(verts, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    el = PlyElement.describe(verts, 'vertex')
    PlyData([el], text=False).write(filename)


def create_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--recorder_path', type=str, default='F:\\recordings\\')
    parser.add_argument('--calibration_path', type=str, default='F:\calibration\calibration\\')
    parser.add_argument('--sample_number', type=int, default=10000)
    parser.add_argument('--save_dir', type=str, default='F:\save_dir\\')
    parser.add_argument('--annotation_path', type=str, default='F:\save_dir\\')
    return parser.parse_args()


def process_dataset(recoder_path, save_dir, cam, annotations):
    depth_dir = osp.join(recoder_path, 'Depth')
    body_index_dir = osp.join(recoder_path, 'BodyIndex')
    skeleton_dir = osp.join(recoder_path, 'Skeleton')
    depth_image_list = glob.glob(depth_dir + '\*' + '.png')
    for depth_path in depth_image_list:
        data_dict = {}
        frame_name = depth_path.split('\\')[-1]
        frame_name = frame_name.replace('.png', '')
        body_index_path = osp.join(body_index_dir, frame_name + '.png')
        depth_image = cv2.imread(depth_path, -1).astype(float)
        body_index_image = cv2.imread(body_index_path, -1).astype(int)
        body_mask = body_index_image == 0
        point_cloud, mask = unproject_depth_image(depth_image, body_mask, cam)
        skeleton_path = osp.join(skeleton_dir, frame_name + '.json')
        save_file_name = osp.join(save_dir, frame_name + '.pkl')
        with open(skeleton_path, 'r') as f:
            skeleton_file = json.load(f)
            joints = skeleton_file['Bodies'][0]['Joints']
        joint_keys = list(joints.keys())
        joints_position = np.zeros((25, 3))
        for index in range(25):
            joint_name = joint_keys[index]
            joints_position[index, :] = joints[joint_name]['Position']
        data_dict['joints'] = joints_position
        data_dict['point_cloud'] = point_cloud
        data_dict['mask'] = mask
        print(save_file_name)
        with open(save_file_name, 'wb') as f:
            pickle.dump(data_dict, f)
        annotations.append(save_file_name)


if __name__ == '__main__':
    args = create_argparse()
    recoder_path = args.recorder_path
    calibration_path = args.calibration_path
    annotation_path = osp.join(args.annotation_path, 'annotation.npy')
    save_dir = args.save_dir
    depth_cam_path = osp.join(calibration_path, 'IR.json')
    with open(depth_cam_path, 'r') as f:
        depth_cam = json.load(f)
    recoder_list = os.listdir(recoder_path)
    annotations = []
    for recoder_name in recoder_list:
        single_recoder_path = osp.join(recoder_path, recoder_name)
        sequence_save_dir = osp.join(save_dir, recoder_name)
        if not osp.exists(sequence_save_dir):
            os.mkdir(sequence_save_dir)
        process_dataset(single_recoder_path, sequence_save_dir, depth_cam, annotations)
    np.save(annotation_path, annotations)
