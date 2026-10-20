"""Four loose camera accessories and a rigid case with a movable lid."""

from __future__ import annotations

import numpy as np
import sapien

from ._base_task import Base_Task
from .utils import create_actor, create_box, create_sapien_urdf_obj, create_table


LEFT_HOME_STATE = [-0.30, 0.20, 0.55, -2.20, 0.0, 2.55, 0.785398]
RIGHT_HOME_STATE = [0.30, 0.20, -0.55, -2.20, 0.0, 2.55, 0.785398]
EQUIPMENT_LAYOUT = {
    "camera": {"x": -0.36, "mass_kg": 0.45},
    "flash": {"x": -0.09, "mass_kg": 0.10},
    "lens": {"x": 0.14, "mass_kg": 0.12},
    "battery": {"x": 0.38, "mass_kg": 0.10},
}
MAX_DEPENETRATION_SPEED_M_S = 0.5
MAX_LINEAR_SPEED_M_S = 0.03
MAX_ANGULAR_SPEED_RAD_S = 0.3
SLOT_POSITION_TOLERANCE_M = 0.020
SLOT_HEIGHT_TOLERANCE_M = 0.008
MAX_ORIENTATION_ERROR_DEG = 25.0


class camera_equipment_packing(Base_Task):
    """Pack all four independent rigid objects into their matching case slots."""

    def setup_demo(self, **kwargs):
        kwargs = kwargs.copy()
        left = kwargs["left_embodiment_config"].copy()
        right = kwargs["right_embodiment_config"].copy()
        left["homestate"] = [LEFT_HOME_STATE.copy(), RIGHT_HOME_STATE.copy()]
        right["homestate"] = [LEFT_HOME_STATE.copy(), RIGHT_HOME_STATE.copy()]
        kwargs["left_embodiment_config"] = left
        kwargs["right_embodiment_config"] = right
        kwargs["table_xy_bias"] = [0.0, -0.10]
        kwargs["table_color_override"] = [0.28, 0.30, 0.32]
        super()._init_task_env_(**kwargs)

    def create_table_and_wall(
        self, table_xy_bias=[0, -0.10], table_height=0.74,
        table_texture_override=None, table_color_override=None,
    ):
        self.table_xy_bias = table_xy_bias
        self.wall_texture = None
        self.table_texture = table_texture_override
        if table_color_override is not None:
            self.table_texture = None
        self.wall = create_box(
            scene=self.scene,
            pose=sapien.Pose([0, 1.2, 1.5]),
            half_size=[3, 0.6, 1.5],
            color=(1, 0.9, 0.9),
            name="wall",
            is_static=True,
        )
        table_height += self.table_z_bias
        self.table = create_table(
            scene=self.scene,
            pose=sapien.Pose([table_xy_bias[0], table_xy_bias[1], table_height]),
            length=1.4,
            width=0.96,
            height=table_height,
            thickness=0.05,
            color=(0.28, 0.30, 0.32) if table_color_override is None else table_color_override,
            is_static=True,
            texture_id=self.table_texture,
        )

    def load_actors(self):
        self.case = create_sapien_urdf_obj(
            scene=self,
            pose=sapien.Pose([0.0, -0.02, 0.741]),
            modelname="camera_packing_case",
            modelid=None,
            fix_root_link=True,
        )
        # ArticulationActor.__init__ overwrites the masses loaded from the file.
        # Restore the body and lid independently after that wrapper is created.
        self.case.set_mass(5.5, links_name=["case_body"])
        self.case.set_mass(1.2, links_name=["case_lid"])
        self.case.set_qpos([np.deg2rad(105)])
        # Zero stiffness leaves a real movable hinge; joint friction parks it open.
        self.case.set_properties(damping=1, stiffness=0, friction=4, force_limit=12)

        for name, layout in EQUIPMENT_LAYOUT.items():
            # Prepared assets already use Z up and bottom-center root origins.
            # The camera faces -Y and the flash faces -X with this identity pose.
            actor = create_actor(
                scene=self,
                pose=sapien.Pose([layout["x"], -0.40, 0.742]),
                modelname=f"camera_packing_{name}",
                convex=True,
                is_static=False,
                model_id=0,
            )
            actor.set_mass(layout["mass_kg"])
            for component in actor.actor.get_components():
                if isinstance(component, sapien.physx.PhysxRigidDynamicComponent):
                    component.max_depenetration_velocity = MAX_DEPENETRATION_SPEED_M_S
            setattr(self, name, actor)

    def play_once(self):
        # Scene construction supplies no expert trajectory or state intervention.
        return self.info

    def check_success(self):
        # These simulator poses belong only to the private evaluator. They are
        # never copied into public observations or presented as discovered goals.
        case_inverse = np.linalg.inv(
            self.case.link_dict["case_body"].get_pose().to_transformation_matrix()
        )
        orientation_cosine = np.cos(np.deg2rad(MAX_ORIENTATION_ERROR_DEG))
        for name in EQUIPMENT_LAYOUT:
            actor = getattr(self, name)
            relative = case_inverse @ actor.get_pose().to_transformation_matrix()
            target = self.case.config["slot_targets"][name]
            offset = relative[:3, 3] - np.asarray(target["position"])
            if (
                np.any(np.abs(offset[:2]) > SLOT_POSITION_TOLERANCE_M)
                or abs(offset[2]) > SLOT_HEIGHT_TOLERANCE_M
            ):
                return False

            target_rotation = sapien.Pose(
                [0, 0, 0], target["quaternion"]
            ).to_transformation_matrix()[:3, :3]
            rotation_error = target_rotation.T @ relative[:3, :3]
            # The round spare lens has no meaningful yaw; keep its axis upright.
            alignment = (
                rotation_error[2, 2] if name == "lens"
                else (np.trace(rotation_error) - 1) / 2
            )
            if alignment < orientation_cosine:
                return False

            body = next(
                component for component in actor.actor.get_components()
                if isinstance(component, sapien.physx.PhysxRigidDynamicComponent)
            )
            if (
                np.linalg.norm(body.linear_velocity) >= MAX_LINEAR_SPEED_M_S
                or np.linalg.norm(body.angular_velocity) >= MAX_ANGULAR_SPEED_RAD_S
            ):
                return False
        return True
