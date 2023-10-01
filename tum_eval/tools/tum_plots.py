import matplotlib.pyplot as plt
import pprint

from evo.core import metrics
from evo.core import sync
from evo.core.sync import TrajectoryPair
from evo.core.trajectory import PosePath3D, PoseTrajectory3D
from evo.tools import plot
from evo.tools.settings import SETTINGS

from tools import tum_tools
from tools.internal.plot_settings import get_figsize

from typing import Dict

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
    

def plot_compare(traj: TrajectoryPair, 
                 align_origin=False,
                 print_stats=False,
                 plot_mode='ape',
                 max_diff=0.01,
                 est_name="Estimated",
                 **kwargs):
    """_summary_

    Args:
        traj (TrajectoryPair): _description_
        align_origin (bool, optional): _description_. Defaults to False.
        print_stats (bool, optional): _description_. Defaults to False.
        plot_error (str, optional): _description_. Defaults to 'ape'.
        max_diff (float, optional): _description_. Defaults to 0.01.
        est_name (str, optional): _description_. Defaults to "Estimated".

    Raises:
        RuntimeError: _description_

    Returns:
        _type_: _description_
    """
    plt.close('all')
    traj_ref, traj_est = traj
    if align_origin:
        traj_est = tum_tools.align_origin(traj_est)
    
    traj_ref, traj_est = tum_tools.sync_trajectories(traj_ref, traj_est, max_diff)
    metric = None
    
    if plot_mode == 'ape':
        metric = plot_ape_errors(traj_ref, traj_est, **kwargs)
    elif plot_mode == 'rpe':
        metric = plot_rpe_errors(traj_ref, traj_est, **kwargs)
    elif plot_mode == 'rpy':        
        plot_rpy_errors(traj_ref, traj_est, est_name)
    elif plot_mode == 'xyz':
        plot_xyz_errors(traj_ref, traj_est, est_name)
    else:
        raise RuntimeError(f"Unsupported plot type: plot_mode={plot_mode}. Must be one of ['ape', 'rpe', 'rpy', 'xyz']!")
    
    if print_stats and metric is not None:
        pprint.pprint(metric.get_all_statistics())
    return metric.get_result()


def compare_plot_multiple(trajecotries, 
                          names=None,
                          plot_mode='xyz',
                          **kwargs):
    plt.close('all')
    if names is not None:
        assert len(names) == len(trajecotries), "Number of given names must be equal to the number of trajecotries"
    else:
        names = [i for i in range(1, len(trajecotries))]

    _, axarr = plt.subplots(3, figsize=get_figsize(wf=1,hf=1))
    
    plot_fn = None
    if plot_mode == 'xyz':
        plot_fn = plot.traj_xyz
    elif plot_mode == 'rpy':
        plot_fn = plot.traj_rpy
    else:
        raise RuntimeError(f"Unsupported Plot mode: plot_mode={plot_mode}. Must be one of ['xyz', 'rpy']")
    
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    for i, (t,n) in enumerate(zip(trajecotries, names)):
        linestyle="-"
        color = colors[i]
        
        if n.lower() == 'reference':
            linestyle = "--"
            color = 'gray'

        plot_fn(axarr, t, linestyle, color, n)
    plt.show()
    

def plot_rpe_errors(traj_ref, traj_est, title=None, pose_relation=metrics.PoseRelation.translation_part):
    rpe_metric = metrics.RPE(pose_relation=pose_relation)
    rpe_metric.process_data((traj_ref, traj_est))
    rpe_stats = rpe_metric.get_all_statistics()
    plot_error_metric(traj_ref, traj_est, rpe_metric, rpe_stats, title)
    return rpe_metric

    
def plot_ape_errors(traj_ref, traj_est, title=None, pose_relation=metrics.PoseRelation.translation_part):
    ape_metric = metrics.APE(pose_relation=pose_relation)
    ape_metric.process_data((traj_ref, traj_est))
    ape_stats = ape_metric.get_all_statistics()
    plot_error_metric(traj_ref, traj_est, ape_metric, ape_stats, title)
    return ape_metric
    

def plot_error_metric(traj_ref, traj_est, metric, stats, title=None):
    plot_mode = plot.PlotMode.xy
    fig = plt.figure(figsize=get_figsize(wf=1, hf=1))
    ax = plot.prepare_axis(fig, plot_mode)
    plot.traj(ax, plot_mode, traj_ref, '--', "gray", "reference")
    plot.traj_colormap(ax, traj_est, metric.error, 
                   plot_mode, min_map=stats["min"], max_map=stats["max"])
    ax.legend()
    if title :
        ax.set_title(title)
    plt.show()


def plot_rpy_errors(traj_ref, traj_est, est_name="Estimated"):
    _, axarr = plt.subplots(3, figsize=get_figsize(wf=1,hf=1))
    
    # Using indices on the x axis instead of timestamps
    traj_ref = PosePath3D(poses_se3=traj_ref.poses_se3)
    traj_est = PosePath3D(poses_se3=traj_est.poses_se3)
    
    plot.traj_rpy(axarr, traj_ref, '--', "gray", "reference")
    plot.traj_rpy(axarr, traj_est, '-', 'tab:blue', est_name)
    plt.show()


def plot_xyz_errors(traj_ref, traj_est, est_name="Estimated"):
    _, axarr = plt.subplots(3, figsize=get_figsize(wf=1,hf=1))

    # Using indices on the x axis instead of timestamps
    traj_ref = PosePath3D(poses_se3=traj_ref.poses_se3)
    traj_est = PosePath3D(poses_se3=traj_est.poses_se3)
    
    plot.traj_xyz(axarr, traj_ref, '--', "gray", "reference")
    plot.traj_xyz(axarr, traj_est, '-', 'tab:blue', est_name)
    plt.show()