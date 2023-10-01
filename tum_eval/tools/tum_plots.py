import matplotlib.pyplot as plt
import pprint

from evo.core import metrics
from evo.core import sync
from evo.core.sync import TrajectoryPair
from evo.core.trajectory import PosePath3D, PoseTrajectory3D
from evo.tools import plot
from evo.tools.settings import SETTINGS

from typing import Dict

from tools.tum_tools import align_origin, sync_trajectories


def plot_trajectories(results: Dict, gt_poses=None, close_all: bool = True, figsize=None) -> None:
    if close_all:
        plt.close("all")
        if figsize is not None:
            fig = plt.figure(f"Trajectory results", figsize=figsize)
        else:
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

    ax.legend(frameon=True, fontsize=17)
    # ax.set_title(f"Sequence {sequence}")
    plt.show()
    
    return fig

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
    

def plot_compare(traj: TrajectoryPair, align_origin=False, print_stats=False, plot_error='ape', max_diff=0.01, est_name="Estimated"):
    
    traj_ref, traj_est = traj
    if align_origin:
        traj_est = align_origin(traj_est)
    
    traj_ref, traj_est = sync_trajectories(traj_ref, traj_est)
    
    ape_metric = metrics.APE()
    ape_metric.process_data((traj_ref, traj_est))
    ape_stats = ape_metric.get_all_statistics()
    
    if print_stats:
        pprint.pprint(ape_stats)
    
    if plot_error == 'ape':
        plot_ape_errors(traj_ref, traj_est, ape_metric, ape_stats)
    elif plot_error == 'rpy':        
        plot_rpy_errors(traj_ref, traj_est, est_name)
    elif plot_error == 'xyz':
        plot_xyz_errors(traj_ref, traj_est, est_name)
    else:
        raise RuntimeError(f"Unsupported plot type: plot_error={plot_error}. Must be one of ['ape', 'rpy', 'xyz']!")
    return ape_stats


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


def plot_rpy_errors(traj_ref, traj_est, est_name="Estimated"):
    _ = plt.figure()
    _, axarr = plt.subplots(3)
    
    # Using indices on the x axis instead of timestamps
    traj_ref = PosePath3D(poses_se3=traj_ref.poses_se3)
    traj_est = PosePath3D(poses_se3=traj_est.poses_se3)
    
    plot.traj_rpy(axarr, traj_ref, '--', "gray", "reference")
    plot.traj_rpy(axarr, traj_est, '-', 'tab:blue', est_name)
    plt.show()


def plot_xyz_errors(traj_ref, traj_est, est_name="Estimated"):
    _ = plt.figure()
    _, axarr = plt.subplots(3)

    # Using indices on the x axis instead of timestamps
    traj_ref = PosePath3D(poses_se3=traj_ref.poses_se3)
    traj_est = PosePath3D(poses_se3=traj_est.poses_se3)
    
    plot.traj_xyz(axarr, traj_ref, '--', "gray", "reference")
    plot.traj_xyz(axarr, traj_est, '-', 'tab:blue', est_name)
    plt.show()