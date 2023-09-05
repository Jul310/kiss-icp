import contextlib
import matplotlib.pyplot as plt
import numpy as np
import os 
import pymap3d as pm


from evo.core import metrics
from evo.core.trajectory import PosePath3D, PoseTrajectory3D
from evo.tools import plot
from evo.tools.settings import SETTINGS
from kiss_icp.datasets import dataset_factory
from kiss_icp.pipeline import OdometryPipeline
from pyquaternion import Quaternion
from typing import Dict


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


def plot_trajectories(results: Dict, gt_poses=None, close_all: bool = True) -> None:
    if close_all:
        plt.close("all")
        fig = plt.figure(f"Trajectory results")
        plot_mode = plot.PlotMode.xy
        ax = plot.prepare_axis(fig, plot_mode)
        
        # Plot GT
        if gt_poses is not None:
            plt.plot(gt_poses[:,0], gt_poses[:,1], c=SETTINGS.plot_reference_color,
                    alpha=SETTINGS.plot_reference_alpha, linestyle='dashed', label="Reference")

    colors = ["red", "green", "blue", "yellow", "orange", "purple", "cyan"]
    for sequence, value in results.items():
        poses = value[2]
        plot.traj(
            ax=ax,
            plot_mode=plot_mode,
            traj=poses,
            label=sequence,
            style=SETTINGS.plot_trajectory_linestyle,
            color=colors.pop(0),
            alpha=SETTINGS.plot_trajectory_alpha,
        )

    ax.legend(frameon=True)
    ax.set_title(f"Sequence {sequence}")
    plt.show()
    
def plot_trajectories_from_poses(traj_ref: PoseTrajectory3D, traj_est: PoseTrajectory3D, close_all: bool = True) -> None:
    if close_all:
        plt.close("all")
        fig = plt.figure(f"Trajectory results")
        plot_mode = plot.PlotMode.xy
        ax = plot.prepare_axis(fig, plot_mode)
        
   
        plot.traj(
            ax=ax,
            plot_mode=plot_mode,
            traj=traj_est,
            label="Estimated",
            style=SETTINGS.plot_trajectory_linestyle,
            color='blue',
            alpha=SETTINGS.plot_trajectory_alpha,
        )
        
        plot.traj(
            ax=ax,
            plot_mode=plot_mode,
            traj=traj_ref,
            label="ground truth",
            style=SETTINGS.plot_reference_linestyle,
            color=SETTINGS.plot_reference_color,
            alpha=SETTINGS.plot_reference_alpha,
        )

    ax.legend(frameon=True)
    # ax.set_title(f"Sequence {sequence}")
    plt.show()
    
    
def plot_ape_errors(traj_ref, traj_est, ape_metric, ape_stats, title=None):
    plot_mode = plot.PlotMode.xy
    fig = plt.figure()
    ax = plot.prepare_axis(fig, plot_mode)
    plot.traj(ax, plot_mode, traj_ref, '--', "gray", "reference")
    plot.traj_colormap(ax, traj_est, ape_metric.error, 
                   plot_mode, min_map=ape_stats["min"], max_map=ape_stats["max"])
    ax.legend()
    ax.set_title(title if title is not None else "")
    plt.show()
    
    
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