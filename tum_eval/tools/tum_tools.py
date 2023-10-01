import contextlib

import math
import numpy as np
import os 
import pymap3d as pm

from evo.core import sync
from evo.core.sync import TrajectoryPair
from evo.core.trajectory import PoseTrajectory3D, PosePath3D

from kiss_icp.datasets import dataset_factory
from kiss_icp.pipeline import OdometryPipeline
from pyquaternion import Quaternion

from scipy.spatial.transform import Rotation as R


def wgsToENU(wgs, origin):
    return np.array(pm.geodetic2enu(*wgs, *origin))


def load_gt_poses(dataset_dir, transform_pos=None):
    """Load GT poses from a list of files. 
    Expect each file to be in the form lat, lon, alt, roll, pitch, yaw
    
    transform_pos can be used to convert to convert the pose
    """
    def _identity(x, *args): 
        return x
    
    if transform_pos is None:
        transform_pos = _identity
        
    gps_dir = os.path.join(dataset_dir, 'oxts', 'data')
    files = []
    for file in os.listdir(gps_dir):
        fpath = os.path.join(gps_dir, file)
        if os.path.isfile(fpath):
            files.append(fpath)
    files.sort()
    
    stamps = np.loadtxt(os.path.join(dataset_dir, "times", "gps_timestamps.txt"))
    timestamps = stamps[:,0] - 1689860000 + stamps[:,1] * 1e-9
            
      
    poses = np.zeros(shape=(len(files), 6))
    origin = None
    for i, file_path in enumerate(files, 0):
        gt_pos = np.genfromtxt(str(file_path), delimiter=" ", dtype=np.float64)
        geo_pos = gt_pos[0:3]
        if origin is None:
            origin = geo_pos
        transformed = np.array([*transform_pos(geo_pos, origin), *gt_pos[3:6]])
        poses[i] = transformed
    return poses[:,:3], timestamps


def save_poses_tum_format(filename, poses, timestamps):
    def _to_tum_format(poses, timestamps):
        tum_data = []
        with contextlib.suppress(ValueError):
            for idx in range(len(poses)):
                tx, ty, tz = poses[idx][:3, -1].flatten()
                qw, qx, qy, qz = Quaternion(matrix=poses[idx], atol=0.01).elements
                tum_data.append([float(timestamps[idx]), tx, ty, tz, qx, qy, qz, qw])
        return np.array(tum_data).astype(np.float64)

    np.savetxt(fname=f"{filename}_tum.txt", X=_to_tum_format(poses, timestamps), fmt="%.4f")


# Aling Origin
def align_origin(traj: TrajectoryPair, gt_orientation_deg=101.0) -> TrajectoryPair:
    yaw = math.pi/2  - (gt_orientation_deg * math.pi/180)
    gt_roation_matrix = R.from_euler('xyz', [0, 0, yaw]).as_matrix()

    origin_se3 = np.eye(4)
    origin_se3[:3,:3] = gt_roation_matrix
    ref_pose_origin = PosePath3D(poses_se3=[origin_se3])
    
    traj_est, traj_ref = traj
    traj_est.align_origin(ref_pose_origin)
    return sync.associate_trajectories(traj_ref, traj_est)

    
def run_kiss_icp_pipeline(dataset_path, config, gt_poses=None):

    data_loader = 'mdv3'
    kiss_dataset = dataset_factory(dataloader=data_loader, data_dir=dataset_path, gt_poses=gt_poses)
    
    # data_loader = 'generic'
    # pcd_dir = os.path.join(dataset_path, "pcd")
    # times_path = os.path.join(dataset_path, 'times', 'lidar_timestamps.txt')
    # kiss_dataset = dataset_factory(dataloader=data_loader, data_dir=pcd_dir)

    pipeline = OdometryPipeline(dataset=kiss_dataset, **config)
    _, result_poses, timestamps = pipeline.run()
    
    trajectory_poses = PoseTrajectory3D(poses_se3=result_poses, timestamps=timestamps)
    return result_poses, trajectory_poses, timestamps