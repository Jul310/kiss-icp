from glob import glob
import numpy as np

from os import path
from pathlib import Path

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
            
if __name__ == "__main__":
    root = "/home/julian/projects/thesis/datasets/ext/converted/mdv3_1/mdv3_full_merge"
    
    pose = path.join(root, "poses", "odometry.kitti")
    pcd = path.join(root, "pcd")
    pcd = Path(pcd)
    kiss_to_interactive_slam(pose, pcd)