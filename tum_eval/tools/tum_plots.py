import folium
import matplotlib.pyplot as plt
import numpy as np
import pprint

from os import path
from copy import deepcopy

from evo.core import metrics
from evo.core import sync
from evo.core.sync import TrajectoryPair
from evo.core.trajectory import PosePath3D, PoseTrajectory3D
from evo.tools import plot
from evo.tools.settings import SETTINGS

from tools import tum_tools
from tools.internal.plot_settings import get_figsize

from typing import Dict, Union, List

def simple_plot(*args, plot_fn, xlabel="", ylabel="", **kwargs):
    plt.close('all')
    f = plt.figure(figsize=get_figsize(wf=1, hf=1))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plot_fn(*args, **kwargs)
    plt.show()
    

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

    
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    
    for i, (sequence, value) in enumerate(results.items()):
        poses = value[2]
        plot.traj(
            ax=ax,
            plot_mode=plot_mode,
            traj=poses,
            label=sequence,
            style=SETTINGS.plot_trajectory_linestyle,
            color=colors[i],
            alpha=SETTINGS.plot_trajectory_alpha,
        )

    ax.legend(frameon=True, fontsize=17)
    # ax.set_title(f"Sequence {sequence}")
    plt.show()

def plot_trajectories_from_poses(traj_ref: PoseTrajectory3D,
                                 traj_est: Union[PoseTrajectory3D, List[PoseTrajectory3D]],
                                 traj_est_label: Union[str, List[str]] = "Estimation",
                                 close_all: bool = True) -> None:
    if close_all:
        plt.close("all")
        
    fig = plt.figure(f"Trajectory results", get_figsize(wf=1, hf=1))
    plot_mode = plot.PlotMode.xy
    ax = plot.prepare_axis(fig, plot_mode)
        
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    
    label_list = []
    if not isinstance(traj_est, list):
        traj_est = list(traj_est)
        
    if isinstance(traj_est_label, list):
        if (len(traj_est_label) == len(traj_est)):
            label_list = traj_est_label
        else:
            raise RuntimeError("Number of estimation trajectories does not match number of labels. Cannot correct this.")
    else:
        label_list = [f"{traj_est_label} {i:02d}" for i in range(1, len(traj_est)+1)]
        
        
    
    for i, traj in enumerate(traj_est, 0):
        plot.traj(
            ax=ax,
            plot_mode=plot_mode,
            traj=traj,
            label=label_list[i],
            style=SETTINGS.plot_trajectory_linestyle,
            color=colors[i],
            alpha=SETTINGS.plot_trajectory_alpha,
        )
    
    plot.traj(ax=ax, plot_mode=plot_mode, traj=traj_ref, label="ground truth", style=SETTINGS.plot_reference_linestyle,
                color=SETTINGS.plot_reference_color, alpha=SETTINGS.plot_reference_alpha)

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
    traj_ref, traj_est = deepcopy(traj)
    if align_origin:
        traj_est = tum_tools.align_origin(traj_est)
        
    if len(traj_ref.poses_se3) != len(traj_est.poses_se3):
        try:
            traj_ref, traj_est = tum_tools.sync_trajectories(traj_ref, traj_est, max_diff)
        except:
            pass
        
    metric = None
    
    if plot_mode == 'ape':
        metric = plot_ape_errors(traj_ref, traj_est, **kwargs)
    elif plot_mode == 'rpe':
        metric = plot_rpe_errors(traj_ref, traj_est, **kwargs)
    elif plot_mode == 'rpy':        
        plot_rpy_errors(traj_ref, traj_est, est_name, **kwargs)
    elif plot_mode == 'xyz':
        plot_xyz_errors(traj_ref, traj_est, est_name, **kwargs)
    else:
        raise RuntimeError(f"Unsupported plot type: plot_mode={plot_mode}. Must be one of ['ape', 'rpe', 'rpy', 'xyz']!")
    
    if print_stats and metric is not None:
        pprint.pprint(metric.get_all_statistics())
    return metric


def compare_plot_multiple(trajecotries, 
                          names=None,
                          plot_mode='xyz',
                          remove_stamps=False,
                          wf=2,
                          hf=.5):
    plt.close('all')
    if names is not None:
        assert len(names) == len(trajecotries), "Number of given names must be equal to the number of trajecotries"
    else:
        names = [i for i in range(1, len(trajecotries))]

    _, axarr = plt.subplots(3, figsize=get_figsize(wf=wf,hf=hf))
    
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
        
        try:
            start = t.timestamps[0]
        except:
            start = None
            t = PosePath3D(poses_se3=t.poses_se3)
            
        if remove_stamps:
            # Using indices on the x axis instead of timestamps
            start = None
            t = PosePath3D(poses_se3=t.poses_se3)
        
        if str(n).lower() == 'reference':
            linestyle = "--"
            color = 'gray'

        plot_fn(axarr, t, linestyle, color, n, start_timestamp=start)
    plt.show()
    

def plot_rpe_errors(traj_ref, traj_est, title=None, pose_relation=metrics.PoseRelation.translation_part, **kwargs):
    rpe_metric = metrics.RPE(pose_relation=pose_relation)
    rpe_metric.process_data((traj_ref, traj_est))
    rpe_stats = rpe_metric.get_all_statistics()
    plot_error_metric(traj_ref, traj_est, rpe_metric, rpe_stats, title)
    return rpe_metric

    
def plot_ape_errors(traj_ref, traj_est, title=None, pose_relation=metrics.PoseRelation.full_transformation, **kwargs):
    ape_metric = metrics.APE(pose_relation=pose_relation)
    ape_metric.process_data((traj_ref, traj_est))
    ape_stats = ape_metric.get_all_statistics()
    plot_error_metric(traj_ref, traj_est, ape_metric, ape_stats, title)
    return ape_metric
    

def plot_error_metric(traj_ref, traj_est, metric, stats, title=None):
    plot_mode = plot.PlotMode.xy
    fig = plt.figure(figsize=get_figsize(wf=0.75, hf=.75))
    # fig = plt.figure()
    ax = plot.prepare_axis(fig, plot_mode)
    plot.traj(ax, plot_mode, traj_ref, '--', "gray", "reference")
    plot.traj_colormap(ax, traj_est, metric.error, 
                   plot_mode, min_map=stats["min"], max_map=stats["max"], plot_start_end_markers=False)
                #    plot_mode, min_map=0, max_map=15, plot_start_end_markers=False)
    ax.legend()
    # cir = plt.Circle((145, -19), 18, color='r',fill=False, linestyle="--")
    # ax.add_patch(cir)
    if title :
        ax.set_title(title)
    plt.show()


def plot_rpy_errors(traj_ref, traj_est, est_name="Estimated",  wf=2, hf=0.5, **kwargs):
    _, axarr = plt.subplots(3, figsize=get_figsize(wf=wf,hf=hf))
    
    # Using indices on the x axis instead of timestamps
    traj_ref = PosePath3D(poses_se3=traj_ref.poses_se3)
    traj_est = PosePath3D(poses_se3=traj_est.poses_se3)
    
    plot.traj_rpy(axarr, traj_ref, '--', "gray", "reference")
    plot.traj_rpy(axarr, traj_est, '-', 'tab:blue', est_name)
    plt.show()


def plot_xyz_errors(traj_ref, traj_est, est_name="Estimated",  wf=2, hf=0.5, **kwargs):
    _, axarr = plt.subplots(3, figsize=get_figsize(wf=wf,hf=hf))

    # Using indices on the x axis instead of timestamps
    traj_ref = PosePath3D(poses_se3=traj_ref.poses_se3)
    traj_est = PosePath3D(poses_se3=traj_est.poses_se3)
    
    plot.traj_xyz(axarr, traj_ref, '--', "gray", "reference")
    plot.traj_xyz(axarr, traj_est, '-', 'tab:blue', est_name)
    plt.show()


def plot_statistical_error_metric(metric, show_plot=False, save_dir="", name_prefix=""):
    _UNIT_LOOKUP = {
        metrics.PoseRelation.full_transformation: "",
        metrics.PoseRelation.translation_part: " $(m)$",
        metrics.PoseRelation.rotation_angle_deg: " $(deg)$",
    }
    
    stats = metric.get_all_statistics()
    metric_type = "APE" if isinstance(metric, metrics.APE) else "RPE"
    unit = _UNIT_LOOKUP[metric.pose_relation]
    
    name = f"{metric_type}_{metric.pose_relation._name_}.pdf"
    
    fig = plt.figure(figsize=get_figsize(wf=1.5, hf=.75))
    plot.error_array(fig.gca(), metric.error,
                 statistics={s:v for s,v in stats.items() if s != "sse"},
                 name=f"{metric_type}{unit}", title=f"{metric_type} w.r.t. " + metric.pose_relation.value, xlabel="index")
    
    if show_plot:
        plt.show()
    
    if save_dir:
        prefix = f"{name_prefix}_" if name_prefix else ""
        save = path.join(save_dir, f"{prefix}{name}")
        plt.savefig(save, format="pdf", bbox_inches="tight")
        
        
def plot_trajectory_segments(traj: PoseTrajectory3D, n=1000, title=None, wf=1, hf=1):
    plot_mode = plot.PlotMode.xy
    fig = plt.figure(figsize=get_figsize(wf=wf, hf=hf))
    # fig = plt.figure()
    ax = plot.prepare_axis(fig, plot_mode)
    
    def split_trajectory(traj: PoseTrajectory3D, num_poses=1000):
        n_max = len(traj.poses_se3)
        n_segments = n_max // num_poses
        n = 0
        result = []
        while n < n_max:
            traj_loop = deepcopy(traj)
            end = n_max if len(result) == n_segments else n + num_poses
            traj_loop.reduce_to_ids(range(n, end))
            result.append(traj_loop)
            n = end
            
        return result
    
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    trajectory_segments = split_trajectory(traj, n)
    
    for i, segment in enumerate(trajectory_segments):  
        color = colors[i%len(colors)] 
        start = segment.positions_xyz[0][0:2]
        end = segment.positions_xyz[-1][0:2]
        
        length = np.linalg.norm(segment.positions_xyz[-1] - segment.positions_xyz[0])
         
        print(f"{i}: Segment from {i*n} to {(i+1)*n} - length: {length}")
        label = f"{i*n} - {(i+1)*n if (i+1)*n < traj.num_poses else traj.num_poses}"
        ax.scatter(*start, marker="o", color=color,
               alpha=1, label=None)
        ax.annotate(label, start)
        # ax.scatter(*end, marker="x", color=color, alpha=1,
        #        label=end_label)

        plot.traj(ax, plot_mode, segment, color=color)
        
    
    # ax.legend()
    if title :
        ax.set_title(title)
    plt.show()
    
    return map