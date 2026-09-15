"""Cafe table-clearing task on RoboTwin's standard table."""

import numpy as np
import sapien

from ._base_task import Base_Task
from .utils import *


LEFT_HOME_STATE = [-0.30, 0.20, 0.55, -2.20, 0.0, 2.55, 0.785398]
RIGHT_HOME_STATE = [0.30, 0.20, -0.55, -2.20, 0.0, 2.55, 0.785398]

WOOD_TABLE_COLOR = (0.42, 0.23, 0.10)
UPRIGHT_CUP_QUAT = (0.5, 0.5, 0.5, 0.5)
TIPPED_CUP_QUAT = (0.0, 2**-0.5, 0.0, 2**-0.5)

BASKET_CENTER_XY = np.asarray([-0.29, 0.13])
BASKET_INTERIOR_HALF_XY = np.asarray([0.075, 0.045])

PAPER_LOBES = (
    ((-0.006, 0.000, 0.000), 0.018),
    ((0.007, 0.003, 0.002), 0.016),
    ((0.000, -0.007, 0.005), 0.015),
    ((0.003, 0.006, -0.004), 0.014),
)

MAX_LINEAR_SPEED_M_S = 0.05
MAX_ANGULAR_SPEED_RAD_S = 0.5
HOME_TOLERANCE_RAD = 0.05
STABLE_SUCCESS_STEPS = 250


def create_paper_wad(*, scene, pose, name):
    """Create one lightweight, irregular, graspable paper-wad actor."""

    native_scene, pose = preprocess(scene, pose)
    builder = native_scene.create_actor_builder()
    builder.set_physx_body_type("dynamic")
    material = sapien.render.RenderMaterial(base_color=[0.93, 0.91, 0.86, 1.0])
    for offset, radius in PAPER_LOBES:
        local_pose = sapien.Pose(offset)
        builder.add_sphere_collision(
            pose=local_pose,
            radius=radius,
            material=native_scene.default_physical_material,
        )
        builder.add_sphere_visual(
            pose=local_pose,
            radius=radius,
            material=material,
        )
    builder.set_initial_pose(pose)
    entity = builder.build(name=name)

    actor_data = {
        "center": [0, 0, 0],
        "extents": [0.026, 0.026, 0.026],
        "scale": [0.026, 0.026, 0.026],
        "target_pose": [np.eye(4).tolist()],
        "contact_points_pose": [np.eye(4).tolist()],
        "functional_matrix": [np.eye(4).tolist()],
        "contact_points_description": ["The exposed surface of the paper wad."],
        "contact_points_group": [[0]],
        "contact_points_mask": [True],
    }
    return Actor(entity, actor_data, mass=0.015)


class cafe_table_reset(Base_Task):
    """Clear two used cups and two paper wads into a small waste basket."""

    def setup_demo(self, **kwargs):
        kwargs = kwargs.copy()
        left_config = kwargs["left_embodiment_config"].copy()
        right_config = kwargs["right_embodiment_config"].copy()
        left_config["homestate"] = [LEFT_HOME_STATE.copy(), RIGHT_HOME_STATE.copy()]
        right_config["homestate"] = [LEFT_HOME_STATE.copy(), RIGHT_HOME_STATE.copy()]
        kwargs["left_embodiment_config"] = left_config
        kwargs["right_embodiment_config"] = right_config
        kwargs["table_color_override"] = WOOD_TABLE_COLOR
        super()._init_task_env_(**kwargs)

    def load_actors(self):
        def actor(
            modelname,
            position,
            *,
            instance_name,
            model_id,
            is_static=False,
            mass=0.05,
            quat=UPRIGHT_CUP_QUAT,
        ):
            result = create_actor(
                scene=self,
                pose=sapien.Pose(position, quat),
                modelname=modelname,
                convex=True,
                model_id=model_id,
                is_static=is_static,
            )
            result.set_name(instance_name)
            if not is_static:
                result.set_mass(mass)
            return result

        self.used_cups = [
            actor(
                "021_cup",
                [-0.20, -0.15, 0.741],
                instance_name="upright_used_cup",
                model_id=6,
            ),
            actor(
                "021_cup",
                [0.14, -0.10, 0.79],
                instance_name="tipped_used_cup",
                model_id=6,
                quat=TIPPED_CUP_QUAT,
            ),
        ]
        self.waste_basket = actor(
            "076_breadbasket",
            [BASKET_CENTER_XY[0], BASKET_CENTER_XY[1], 0.741],
            instance_name="waste_basket",
            model_id=0,
            is_static=True,
        )
        self.cafe_plant = actor(
            "120_plant",
            [0.34, 0.22, 0.741],
            instance_name="cafe_plant",
            model_id=0,
            is_static=True,
        )
        self.paper_wads = [
            create_paper_wad(
                scene=self,
                pose=sapien.Pose([-0.02, -0.17, 0.77]),
                name="paper_wad_1",
            ),
            create_paper_wad(
                scene=self,
                pose=sapien.Pose([0.04, -0.02, 0.77]),
                name="paper_wad_2",
            ),
        ]

        self._stable_success_steps = 0
        self.subgoal_vector = [False] * 4

    def _transfer(self, item, arm_tag, target_pose, *, pre_dis=0.10):
        self.move(self.grasp_actor(item, arm_tag=arm_tag, pre_grasp_dis=0.08))
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.10, move_axis="arm"))
        self.move(
            self.place_actor(
                item,
                arm_tag=arm_tag,
                target_pose=target_pose,
                constrain="free",
                pre_dis=pre_dis,
            )
        )
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.10, move_axis="arm"))
        self.move(self.back_to_origin(arm_tag=arm_tag))

    def play_once(self):
        """Diagnostic native expert; never use this as public-observation evidence."""

        basket_z = float(self.waste_basket.get_pose().p[2] + 0.055)
        targets = (*self.used_cups, *self.paper_wads)
        placements = (
            [-0.325, 0.13, basket_z],
            [-0.255, 0.13, basket_z],
            [-0.29, 0.105, basket_z + 0.035],
            [-0.29, 0.155, basket_z + 0.035],
        )
        for target, arm_tag, position in zip(
            targets,
            ("left", "right", "left", "right"),
            placements,
        ):
            self._transfer(
                target,
                arm_tag,
                sapien.Pose(position, UPRIGHT_CUP_QUAT),
            )

        self.info["info"] = {
            "{upright_used_cup}": "021_cup/base6",
            "{tipped_used_cup}": "021_cup/base6",
            "{paper_wad_1}": "procedural_paper_wad",
            "{paper_wad_2}": "procedural_paper_wad",
            "task_id": "cafe_table_clearing_v2",
        }
        return self.info

    @staticmethod
    def _inside(actor, center, half_extent):
        position = np.asarray(actor.get_pose().p[:2], dtype=float)
        return bool(np.all(np.abs(position - center) <= half_extent))

    @staticmethod
    def _body_speeds(actor):
        bodies = [
            component
            for component in actor.actor.get_components()
            if hasattr(component, "linear_velocity")
            and hasattr(component, "angular_velocity")
        ]
        if len(bodies) != 1:
            raise RuntimeError(f"dynamic body unavailable for {actor.get_name()}")
        linear = float(np.linalg.norm(np.asarray(bodies[0].linear_velocity, dtype=float)))
        angular = float(np.linalg.norm(np.asarray(bodies[0].angular_velocity, dtype=float)))
        return linear, angular

    def _stable(self, actor):
        linear, angular = self._body_speeds(actor)
        return linear < MAX_LINEAR_SPEED_M_S and angular < MAX_ANGULAR_SPEED_RAD_S

    def _unheld(self, actor):
        return not self.get_gripper_actor_contact_position(actor.get_name())

    def _supported(self, actor, support_name):
        return bool(self.check_actors_contact(actor.get_name(), support_name))

    def _target_in_basket(self, target):
        return bool(
            self._inside(target, BASKET_CENTER_XY, BASKET_INTERIOR_HALF_XY)
            and self._supported(target, self.waste_basket.get_name())
            and self._stable(target)
            and self._unheld(target)
        )

    def _arms_home(self):
        measured = (
            np.asarray(self.robot.left_entity.get_qpos()[:7], dtype=float),
            np.asarray(self.robot.right_entity.get_qpos()[:7], dtype=float),
        )
        expected = (np.asarray(LEFT_HOME_STATE), np.asarray(RIGHT_HOME_STATE))
        return all(
            values.shape == target.shape
            and np.max(np.abs(values - target)) <= HOME_TOLERANCE_RAD
            for values, target in zip(measured, expected)
        )

    def check_success(self):
        self.subgoal_vector = [
            self._target_in_basket(target)
            for target in (*self.used_cups, *self.paper_wads)
        ]
        final_state = bool(
            all(self.subgoal_vector)
            and self.robot.is_left_gripper_open()
            and self.robot.is_right_gripper_open()
            and self._arms_home()
        )
        self._stable_success_steps = (
            self._stable_success_steps + 1 if final_state else 0
        )
        return self._stable_success_steps >= STABLE_SUCCESS_STEPS
