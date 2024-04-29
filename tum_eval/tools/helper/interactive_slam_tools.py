from glob import glob
import numpy as np

from os import path
from pathlib import Path

from evo.tools.file_interface import read_tum_trajectory_file

def kiss_to_interactive_slam(pose_file, pcd_dir):
    with open(pose_file, 'r') as f:
        lines = np.loadtxt(f, dtype=np.float64)
    pcd_files = list(sorted(pcd_dir.glob("*.pcd")))
    
    for line, pcd_file in zip(lines, pcd_files):
        out = path.join("tmp", pcd_file.stem + ".odom")
        pose_matrix = np.reshape(line, (3,4))
        pose_matrix = np.vstack((pose_matrix, [0,0,0,1]))
        with open(out, 'w') as f:
            np.savetxt(f, pose_matrix, delimiter=' ')
            
def kiss_to_odom_files(pose_file, pcd_dir):
    with open(pose_file, 'r') as f:
        lines = np.loadtxt(f, dtype=np.float64)
    pcd_files = list(sorted(pcd_dir.glob("*.pcd")))
    
    assert (len(lines) == len(pcd_files))
    
    for line, pcd_file in zip(lines, pcd_files):
        out = path.join("tmp", pcd_file.stem + ".odom")
        pose_matrix = np.reshape(line, (3,4))
        pose_matrix = np.vstack((pose_matrix, [0,0,0,1]))
        with open(out, 'w') as f:
            np.savetxt(f, pose_matrix, delimiter=' ')
            
def tum_to_odom_files(pose_file, pcd_dir):
    traj = read_tum_trajectory_file(pose_file)
    pcd_files = list(sorted(pcd_dir.glob("*.pcd")))
    
    assert (traj.num_poses == len(pcd_files))
    
    for pose, pcd_file in zip(traj.poses_se3, pcd_files):
        out = path.join(pcd_dir, pcd_file.stem + ".odom")
        with open(out, 'w') as f:
            np.savetxt(f, pose, delimiter=' ')
            
if __name__ == "__main__":
    # root = "/home/julian/projects/thesis/datasets/ext/converted/mdv3_1/mdv3_full_merge"
    root = "/home/julian/projects/thesis/datasets/converted/mdv3_1/mdv3_sync_merge"
    
    in_path = "/home/julian/projects/thesis/datasets/converted/mdv3_1/mdv3_full_merge/poses/raw_kiss_processed_pcd.tum"
    pcd_path = "/home/julian/projects/thesis/datasets/mapping_analysis/submaps_075_optimized/submaps_raw/"
    
    tum_to_odom_files(in_path, Path(pcd_path))
    
    # pose = path.join(root, "poses", "raw_kiss.tum")
    # pcd = path.join(root, "pcd")
    # pcd = Path(pcd)
    # kiss_to_interactive_slam(pose, pcd)