import json
import os

def tum_to_json(in_file, out_file):
    with open(in_file, 'r') as f:
        tum = f.readlines()
        tum = [tuple(line.strip().split()) for line in tum]
    
    # tum format is T x y z qx qy qz qw
    # out format is 
    # x y z qw qx qy qz
    tum = [(t[1], t[2], t[3], t[7], t[4], t[5], t[6]) for t in tum]
    
    with open(out_file, 'w') as f:
        lines = [" ".join(l) for l in tum]
        f.write('\n'.join(lines) + '\n')
        

if __name__ == "__main__":
    in_file = "/home/julian/projects/thesis/datasets/converted/mdv3_2/mdv3_full_merge/poses/odometry.tum"
    out_file = "/home/julian/projects/thesis/datasets/ext/converted/mdv3_2/mdv3_full_merge/pose.json"
    tum_to_json(in_file, out_file)
    
    # pcd_num = 0
    # pcd_dir = "/home/julian/projects/thesis/datasets/ext/converted/mdv3_2/mdv3_full_merge/pcd/"
    # for file in sorted(os.listdir(pcd_dir)):
    #     if not file.endswith('.pcd'):
    #         continue
        
        
    #     src = os.path.join(pcd_dir, file)
    #     dst = os.path.join(pcd_dir, f"{pcd_num:05d}.pcd")
    #     os.rename(src, dst)
    #     pcd_num+=1
        
    