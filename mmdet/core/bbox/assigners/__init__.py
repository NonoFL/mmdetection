from .approx_max_iou_assigner import ApproxMaxIoUAssigner
from .assign_result import AssignResult
from .atss_assigner import ATSSAssigner
from .base_assigner import BaseAssigner
from .center_region_assigner import CenterRegionAssigner
from .grid_assigner import GridAssigner
from .hungarian_assigner import HungarianAssigner
from .max_iou_assigner import MaxIoUAssigner
from .point_assigner import PointAssigner
from .region_assigner import RegionAssigner
from .sim_ota_assigner import SimOTAAssigner
from .uniform_assigner import UniformAssigner
from .abs_assigner import ABSAssigner
from .bs_assigner import BSAssigner
from .atss_assigner_img import ATSSAssigner_img
from .atss_assignerv3 import ATSSAssignerv3
from .atss_assignerv4 import ATSSAssignerv4
from .atss_assignerv5 import ATSSAssignerv5
from .atss_assignerv6 import ATSSAssignerv6
from .atss_assignerv7 import ATSSAssignerv7
from .atss_assignerv8 import ATSSAssignerv8
from .atss_assignerv9 import ATSSAssignerv9
from .atss_assignerv10 import ATSSAssignerv10
from .atss_assignerv11 import ATSSAssignerv11
from .new_assigner import NewAssigner
__all__ = [
    'BaseAssigner', 'MaxIoUAssigner', 'ApproxMaxIoUAssigner', 'AssignResult',
    'PointAssigner', 'ATSSAssigner', 'CenterRegionAssigner', 'GridAssigner',
    'HungarianAssigner', 'RegionAssigner', 'UniformAssigner', 'SimOTAAssigner',
    'ABSAssigner', 'BSAssigner', 'ATSSAssigner_img', 'ATSSAssignerv3','ATSSAssignerv4', 'ATSSAssignerv5',
    'NewAssigner', 'ATSSAssignerv6','ATSSAssignerv7', 'ATSSAssignerv8', 'ATSSAssignerv9', 'ATSSAssignerv10',
    'ATSSAssignerv11'
]
