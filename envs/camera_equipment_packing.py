"""Bright camera-rental counter with four packable items and an open case."""

from __future__ import annotations

import numpy as np
import sapien

from ._base_task import Base_Task
from .camera_rental_decor import create_front_image_panel
from .utils import create_actor, create_box, create_sapien_urdf_obj


# Retract both wrists to the sides of the case's entire lid sweep. These values
# enter the normal Robot.move_to_homestate path before actors are loaded.
LEFT_HOME_STATE = [0.0, -np.pi / 4, 0.0, -3 * np.pi / 4, 0.0, np.pi / 2, np.pi / 4]
RIGHT_HOME_STATE = [0.0, -np.pi / 4, 0.0, -3 * np.pi / 4, 0.0, np.pi / 2, np.pi / 4]
# The native two-arm loader still adds -/+0.30 m in world X. These inputs
# therefore create bases at (-0.84, -0.10, 0.75) and (+0.84, -0.10, 0.75),
# facing inward along +X and -X respectively. The assembly identity is unchanged.
SIDE_ARM_LOADER_POSES = [
    [-0.54, -0.10, 0.75, 1.0, 0.0, 0.0, 0.0],
    [0.54, -0.10, 0.75, 0.0, 0.0, 0.0, 1.0],
]
COUNTER_DIMS_M = [1.40, 0.96, 0.74]
ARM_PEDESTAL_CENTERS_X_M = [-0.84, 0.84]
PREP_TRAY_ROOT_M = [0.0, -0.397, 0.740]
STORAGE_RACK_ROOT_M = [0.0, 0.625, 0.0]
# Centers are physical XY bounding-box centers, not asset root translations.
EQUIPMENT_LAYOUT = {
    "camera": {"center_xy_m": [-0.315, -0.400], "support_z_m": 0.747, "mass_kg": 0.45},
    "flash": {"center_xy_m": [-0.120, -0.385], "support_z_m": 0.747, "mass_kg": 0.10},
    "lens": {"center_xy_m": [0.140, -0.370], "support_z_m": 0.747, "mass_kg": 0.12},
    "battery": {"center_xy_m": [0.310, -0.415], "support_z_m": 0.747, "mass_kg": 0.10},
}
INITIAL_SUPPORT_CLEARANCE_M = 0.002
WALL_FRONT_Y_M = 0.72
POSTER_CENTERS_M = [[-0.46, 0.702, 1.52], [0.46, 0.702, 1.52]]
POSTER_TEXTURE_IDS = ["rental_poster_v2_en", "rental_ratecard_v4_en"]
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
        left["robot_pose"] = [pose.copy() for pose in SIDE_ARM_LOADER_POSES]
        right["robot_pose"] = [pose.copy() for pose in SIDE_ARM_LOADER_POSES]
        kwargs["left_embodiment_config"] = left
        kwargs["right_embodiment_config"] = right
        kwargs["table_xy_bias"] = [0.0, -0.10]
        super()._init_task_env_(**kwargs)

    def create_table_and_wall(
        self, table_xy_bias=[0, -0.10], table_height=0.74,
        table_texture_override=None, table_color_override=None,
    ):
        self.table_xy_bias = table_xy_bias
        self.wall_texture = None
        # The fixed asset owns the wood material, rounded top and cabinet body.
        self.table_texture = None
        self.wall = create_box(
            scene=self.scene,
            pose=sapien.Pose([0, WALL_FRONT_Y_M + 0.05, 1.30]),
            half_size=[1.6, 0.05, 1.30],
            color=(0.91, 0.93, 0.93), name="rental_warm_white_wall", is_static=True,
        )
        self.table = create_actor(
            scene=self.scene,
            pose=sapien.Pose([*table_xy_bias, self.table_z_bias]),
            modelname="camera_rental_counter", model_id=0, convex=True, is_static=True,
        )
        self.arm_pedestals = [
            create_actor(
                scene=self.scene,
                pose=sapien.Pose([x, -0.10, self.table_z_bias]),
                modelname="camera_rental_arm_pedestal", model_id=0,
                convex=True, is_static=True,
            )
            for x in ARM_PEDESTAL_CENTERS_X_M
        ]
        # Conservative box of the entire rounded cabinet, expressed in world
        # coordinates. The existing planner transforms it into each arm frame.
        self.planning_table = {
            "dims": COUNTER_DIMS_M.copy(),
            "pose": [*table_xy_bias, 0.37 + self.table_z_bias, 1, 0, 0, 0],
        }
        self._create_rental_counter_decor()

    def _create_rental_counter_decor(self):
        """Keep all decorative solids fixed and outside the loose-item workspace."""
        navy = (0.055, 0.115, 0.19)
        self.rental_decor = []

        def box(name, position, half_size, color, texture_id=None):
            position = np.asarray(position, dtype=float).copy()
            position[2] += self.table_z_bias
            actor = create_box(
                scene=self.scene, pose=sapien.Pose(position), half_size=half_size,
                color=color, is_static=True, name=name, texture_id=texture_id,
            )
            self.rental_decor.append(actor)
            return actor

        # Both asset roots are their bottom centers; their geometry is already metric.
        # The rack owns its fixed spare cases, with no duplicate loose task objects.
        for attribute, modelname, root in (
            ("prep_tray", "camera_rental_prep_tray", PREP_TRAY_ROOT_M),
            ("storage_rack", "camera_rental_storage_rack", STORAGE_RACK_ROOT_M),
        ):
            position = np.asarray(root, dtype=float).copy()
            position[2] += self.table_z_bias
            actor = create_actor(
                scene=self.scene, pose=sapien.Pose(position), modelname=modelname,
                model_id=0, convex=True, is_static=True,
            )
            setattr(self, attribute, actor)
            self.rental_decor.append(actor)

        for index, (center, texture_id) in enumerate(zip(POSTER_CENTERS_M, POSTER_TEXTURE_IDS)):
            x, y, z = center
            box(f"rental_poster_frame_{index}", [x, y + 0.008, z], [0.162, 0.008, 0.212], navy)
            self.rental_decor.append(create_front_image_panel(
                self.scene, sapien.Pose([x, y - 0.001, z + self.table_z_bias]),
                width=0.30, height=0.40, texture_id=texture_id,
                name=f"rental_photography_poster_{index}",
            ))
        box("rental_counter_sign_backing", [0, -0.554, 0.52], [0.154, 0.001, 0.030], navy)
        self.rental_decor.append(create_front_image_panel(
            self.scene, sapien.Pose([0, -0.5555, 0.52 + self.table_z_bias]),
            width=0.30, height=0.056, texture_id="rental_counter_sign_v2",
            name="rental_counter_sign",
        ))

    def load_actors(self):
        self.case = create_sapien_urdf_obj(
            scene=self,
            pose=sapien.Pose([0.0, 0.10, 0.741]),
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
            # The source root is not the rotated support point. Use the prepared
            # collision bounds to align physical XY center and bottom to the support.
            prepared = self.case.config["equipment_poses"][name]
            support_origin = np.asarray([
                *layout["center_xy_m"], layout["support_z_m"] + INITIAL_SUPPORT_CLEARANCE_M
            ])
            root = support_origin + np.asarray(prepared["root_offset_from_xy_center_and_bottom_m"])
            actor = create_actor(
                scene=self,
                pose=sapien.Pose(root, prepared["quaternion_wxyz"]),
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
