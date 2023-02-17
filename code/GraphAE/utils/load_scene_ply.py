import os
import sys

rootPath = '../'
sys.path.append(rootPath)

import os.path as osp
import cv2
import numpy as np
import json
import trimesh
import argparse
# os.environ["PYOPENGL_PLATFORM"] = "egl"
# os.environ["PYOPENGL_PLATFORM"] = "osmesa"
import pyrender
import PIL.Image as pil_img
import pickle
import smplx
import torch
from tqdm import tqdm
from os.path import basename
from argparse import Namespace

from utils import *
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def scene_to_trimeshes(recording_name, scene_name, frame_id):
    args = Namespace(
        model_folder='/home/robert/g/EgoBody/models',
        model_type='smplx',
        num_pca_comps=12,
        recording_name=recording_name,
        release_data_root='/home/robert/g/EgoBody/dataset',
        rendering_mode='3d',
        save_undistorted_img=False,
        scale=4,
        scene_name=scene_name,
        start=0,
        step=50,
        view='master'
    )
    calib_trans_dir = os.path.join(args.release_data_root, 'calibrations', args.recording_name)  # extrinsics
    camera_params_dir = os.path.join(args.release_data_root, 'kinect_cam_params')  # intrinsics
    depth_dir = osp.join(args.release_data_root, 'kinect_depth', args.recording_name, args.view)

    data_split_info = pd.read_csv(os.path.join(args.release_data_root, 'data_splits.csv'))
    train_split_list = list(data_split_info['train'])
    val_split_list = list(data_split_info['val'])
    test_split_list = list(data_split_info['test'])
    if args.recording_name in train_split_list:
        split = 'train'
    elif args.recording_name in val_split_list:
        split = 'val'
    elif args.recording_name in test_split_list:
        split = 'test'
    else:
        print('Error: {} not in all splits.'.format(args.recording_name))
        exit()

    if args.model_type == 'smplx':
        fitting_root_interactee = osp.join(args.release_data_root, 'smplx_interactee_{}'.format(split), args.recording_name)
        fitting_root_camera_wearer = osp.join(args.release_data_root, 'smplx_camera_wearer_{}'.format(split), args.recording_name)
    elif args.model_type == 'smpl':
        fitting_root_interactee = osp.join(args.release_data_root, 'smpl_interactee_{}'.format(split), args.recording_name)
        fitting_root_camera_wearer = osp.join(args.release_data_root, 'smpl_camera_wearer_{}'.format(split), args.recording_name)
    else:
        print('Error: body model type error!')
        exit()


    ########## load calibration from sub kinect to main kinect (between color cameras)
    # master: kinect 12, sub_1: kinect 11, sub_2: kinect 13, sub_3, kinect 14, sub_4: kinect 15
    if args.view == 'sub_1':
        trans_subtomain_path = osp.join(calib_trans_dir, 'cal_trans', 'kinect_11to12_color.json')
    elif args.view == 'sub_2':
        trans_subtomain_path = osp.join(calib_trans_dir, 'cal_trans', 'kinect_13to12_color.json')
    elif args.view == 'sub_3':
        trans_subtomain_path = osp.join(calib_trans_dir, 'cal_trans', 'kinect_14to12_color.json')
    elif args.view == 'sub_4':
        trans_subtomain_path = osp.join(calib_trans_dir, 'cal_trans', 'kinect_15to12_color.json')
    if args.view != 'master':
        if not os.path.exists(trans_subtomain_path):
            print('[ERROR] {} does not have the view of {}!'.format(args.recording_name, args.view))
            exit()
        with open(osp.join(trans_subtomain_path), 'r') as f:
            trans_subtomain = np.asarray(json.load(f)['trans'])
            trans_maintosub = np.linalg.inv(trans_subtomain)


    ################################################ read body idx info
    df = pd.read_csv(os.path.join(args.release_data_root, 'data_info_release.csv'))  # todo
    recording_name_list = list(df['recording_name'])
    start_frame_list = list(df['start_frame'])
    end_frame_list = list(df['end_frame'])
    body_idx_fpv_list = list(df['body_idx_fpv'])
    gender_0_list = list(df['body_idx_0'])
    gender_1_list = list(df['body_idx_1'])

    body_idx_fpv_dict = dict(zip(recording_name_list, body_idx_fpv_list))
    gender_0_dict = dict(zip(recording_name_list, gender_0_list))
    gender_1_dict = dict(zip(recording_name_list, gender_1_list))
    start_frame_dict = dict(zip(recording_name_list, start_frame_list))
    end_frame_dict = dict(zip(recording_name_list, end_frame_list))

    ######## get body idx for camera wearer/second person
    interactee_idx = int(body_idx_fpv_dict[args.recording_name].split(' ')[0])
    camera_wearer_idx = 1 - interactee_idx
    ######### get gender for camera weearer/second person
    interactee_gender = body_idx_fpv_dict[args.recording_name].split(' ')[1]
    if camera_wearer_idx == 0:
        camera_wearer_gender = gender_0_dict[args.recording_name].split(' ')[1]
    elif camera_wearer_idx == 1:
        camera_wearer_gender = gender_1_dict[args.recording_name].split(' ')[1]

    ###########################################
    if args.rendering_mode == '3d' or args.rendering_mode == 'both':
        scene_dir = os.path.join(os.path.join(args.release_data_root, 'scene_mesh'), args.scene_name)
        static_scene = trimesh.load(osp.join(scene_dir, args.scene_name + '.obj'))
        cam2world_dir = os.path.join(calib_trans_dir, 'cal_trans/kinect12_to_world')  # transformation from master camera to scene mesh
        with open(os.path.join(cam2world_dir, args.scene_name + '.json'), 'r') as f:
            trans = np.array(json.load(f)['trans'])
        trans = np.linalg.inv(trans)
        static_scene.apply_transform(trans)
        if args.view != 'master':
            static_scene.apply_transform(trans_maintosub)

    ########## read kinect color camera intrinsics
    with open(osp.join(camera_params_dir, 'kinect_{}'.format(args.view), 'Color.json'), 'r') as f:
        color_cam = json.load(f)
    [f_x, f_y] = color_cam['f']
    [c_x, c_y] = color_cam['c']

    ########## create render camera
    camera_pose = np.eye(4)
    camera_pose = np.array([1.0, -1.0, -1.0, 1.0]).reshape(-1, 1) * camera_pose
    camera = pyrender.camera.IntrinsicsCamera(
        fx=f_x, fy=f_y,
        cx=c_x, cy=c_y)
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=4.0)

    material_camera_wearer = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.0,
        alphaMode='OPAQUE',
        baseColorFactor=(70 / 255, 130 / 255, 180 / 255, 1.0)  # blue, camera_wearer
    )

    ######## create smplx/smpl body models
    if args.model_type == 'smplx':
        model_camera_wearer = smplx.create(args.model_folder, model_type='smplx',
                                           gender=camera_wearer_gender, ext='npz', num_pca_comps=args.num_pca_comps,
                                           create_global_orient=True, create_transl=True, create_body_pose=True,
                                           create_betas=True,
                                           create_left_hand_pose=True, create_right_hand_pose=True,
                                           create_expression=True, create_jaw_pose=True, create_leye_pose=True,
                                           create_reye_pose=True).to(device)
    elif args.model_type == 'smpl':
        model_camera_wearer = smplx.create(args.model_folder, model_type='smpl', gender=camera_wearer_gender).to(device)

    ##### read camera wearer smplx params
    with open(osp.join(fitting_root_camera_wearer, 'body_idx_{}'.format(camera_wearer_idx), 'results', frame_id, '000.pkl'), 'rb') as f:
        param = pickle.load(f)
    torch_param = {}
    if args.model_type == 'smpl':
        torch_param['transl'] = torch.tensor(param['transl']).to(device)
        torch_param['global_orient'] = torch.tensor(param['global_orient']).to(device)
        torch_param['betas'] = torch.tensor(param['betas']).to(device)
        torch_param['body_pose'] = torch.tensor(param['body_pose']).to(device)
    elif args.model_type == 'smplx':
        for key in param.keys():
            if key in ['pose_embedding', 'camera_rotation', 'camera_translation', 'gender']:
                continue
            else:
                torch_param[key] = torch.tensor(param[key]).to(device)

    output = model_camera_wearer(return_verts=True, **torch_param)
    vertices = output.vertices.detach().cpu().numpy().squeeze()

    body = trimesh.Trimesh(vertices, model_camera_wearer.faces, process=False)
    if args.view != 'master':
        body.apply_transform(trans_maintosub)
    body_mesh_camera_wearer = pyrender.Mesh.from_trimesh(body, material=material_camera_wearer)
    static_scene_mesh = pyrender.Mesh.from_trimesh(static_scene)

    return {'body': body, 'scene': static_scene_mesh, 'camera': camera, 'camera_pose': camera_pose}


    ###### render in 3d scene
    if args.rendering_mode == '3d' or args.rendering_mode == 'both':
        static_scene_mesh = pyrender.Mesh.from_trimesh(static_scene)

        # Generate scene
        scene = pyrender.Scene()
        scene.add(camera, pose=camera_pose)
        scene.add(light, pose=camera_pose)
        scene.add(static_scene_mesh, 'mesh')

        r = pyrender.OffscreenRenderer(viewport_width=1920,
                                        viewport_height=1080)
        color, depth = r.render(scene)
        color = color.astype(np.float32) / 255.0
        img = pil_img.fromarray((255 * depth/depth.max()).astype(np.uint8))
        img = img.resize((int(1920 / args.scale), int(1080 / args.scale)))
        img.save(os.path.join(args.save_root, f'{args.scene_name}_{args.recording_name}_{frame_id}_depth_vis.jpg'))

        # Generate silhouette
        scene = pyrender.Scene()
        scene.add(camera, pose=camera_pose)
        scene.add(light, pose=camera_pose)
        scene.add(body_mesh_camera_wearer, 'body_mesh_camera_wearer')
        r = pyrender.OffscreenRenderer(viewport_width=1920,
                                        viewport_height=1080)
        _, depth = r.render(scene)
        silhouette = (depth > 0).astype(np.float32)
        img = pil_img.fromarray((silhouette * 255).astype(np.uint8))
        img = img.resize((int(1920 / args.scale), int(1080 / args.scale)))
        img.save(os.path.join(args.save_root, f'{img_name}_silhouette.jpg'))

        # Generate both for reference
        scene = pyrender.Scene()
        scene.add(camera, pose=camera_pose)
        scene.add(light, pose=camera_pose)
        # scene.add(body_mesh_camera_wearer, 'body_mesh_camera_wearer')
        scene.add(static_scene_mesh, 'mesh')
        r = pyrender.OffscreenRenderer(viewport_width=224 * 7,
                                        viewport_height=224 * 7)
        color, depth = r.render(scene)
        silhouette = (depth > 0).astype(np.float32)
        color = color.astype(np.float32) / 255.0
        img = pil_img.fromarray((color * 255).astype(np.uint8))
        img = img.resize((int(224 * 7 / args.scale), int(224 * 7 / args.scale)))
        img.save(os.path.join(args.save_root, f'{img_name}_scene.jpg'))

        body_ball = pyrender.Mesh.from_trimesh(trimesh.primitives.Sphere(radius=0.1, center=(x,y,z)))

        # Scene with ball
        scene = pyrender.Scene()
        scene.add(camera, pose=camera_pose)
        scene.add(light, pose=camera_pose)
        scene.add(body_ball, 'body_ball')
        scene.add(static_scene_mesh, 'mesh')
        r = pyrender.OffscreenRenderer(viewport_width=1920,
                                        viewport_height=1080)
        color, depth = r.render(scene)
        silhouette = (depth > 0).astype(np.float32)
        color = color.astype(np.float32) / 255.0
        img = pil_img.fromarray((color * 255).astype(np.uint8))
        img = img.resize((int(1920 / args.scale), int(1080 / args.scale)))
        img.save(os.path.join(args.save_root, f'{img_name}_ball.jpg'))

    return {'img_name': img_names, 'x': xs, 'y': ys, 'z': zs}