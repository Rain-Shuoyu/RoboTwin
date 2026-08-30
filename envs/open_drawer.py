"""Generic articulated drawer task for OpenSkillBench.

The task exposes native scene state and a success predicate only.  Agentic
grounding and motion planning remain in OpenSkillBench; this module does not
contain a privileged interaction routine.
"""

from __future__ import annotations

import numpy as np

from ._base_task import Base_Task
from .utils import rand_create_sapien_urdf_obj


class open_drawer(Base_Task):
    """One closed prismatic drawer."""

    def setup_demo(self, **kwargs):
        super()._init_task_env_(**kwargs, table_static=False)
        self.cabinet.set_qpos(np.zeros_like(self.cabinet.get_qpos()))
        self.scene.step()
        self._update_render()

    def load_actors(self):
        self.cabinet = rand_create_sapien_urdf_obj(
            scene=self,
            modelname="036_cabinet",
            modelid=1,
            xlim=[0.0, 0.0],
            ylim=[0.155, 0.155],
            rotate_rand=False,
            rotate_lim=[0.0, 0.0, 0.0],
            qpos=[1.0, 0.0, 0.0, 1.0],
            fix_root_link=True,
        )
        self.add_prohibit_area(self.cabinet, padding=0.01)

    def articulation_descriptor(self):
        """Return geometry needed by the native articulated adapter."""

        joint = self.cabinet.actor.get_active_joints()[0]
        joint_pose = joint.get_global_pose().to_transformation_matrix()
        axis = joint_pose[:3, 0]
        limits = np.asarray(self.cabinet.get_qlimits(), dtype=np.float64)
        return {
            "motion_type": "prismatic",
            "open_direction_world": axis.tolist(),
            "max_travel_m": float(limits[0, 1] - limits[0, 0]),
        }

    def check_success(self):
        """Return true after the drawer travels at least 25% of its range."""

        qpos = np.asarray(self.cabinet.get_qpos(), dtype=np.float64)
        limits = np.asarray(self.cabinet.get_qlimits(), dtype=np.float64)
        if qpos.size == 0 or limits.size == 0:
            return False
        lower, upper = limits[0]
        target = lower + 0.25 * (upper - lower)
        return bool(qpos[0] >= target) if upper >= lower else bool(qpos[0] <= target)
