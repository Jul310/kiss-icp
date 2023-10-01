from enum import Enum

COLOR_TUM_BLUE = "#0065BD"
class Mdv3Dataset(Enum):
    FULL_MERGE = "mdv3_full_merge"
    FULL_NO_MERGE = "mdv3_full_no_merge"
    INNOVUSION_NO_MERGE = "mdv3_innovusion_no_merge"
    INNOVUSION_MERGE = "mdv3_innovusion_merge"
    OUSTER_NO_MERGE = "mdv3_ouster_no_merge"
    OUSTER_MERGE = "mdv3_ouster_merge"
    OUSTER_INNOVUSION_FRONT_NO_MERGE = "mdv3_ouster_inno_front_no_merge"
    SYNC_MERGE = "mdv3_sync_merge"
    
KISS_CONFIG = {
    'config': None,
    'deskew': False,
    'max_range': None,
    'visualize': False,
    'n_scans': -1,
    'jump': 0,
    'out_dir': "data/kiss_results"
}