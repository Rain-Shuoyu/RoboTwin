"""Cafe table-reset task on RoboTwin's standard table."""

import numpy as np
import sapien

from ._base_task import Base_Task
from .utils import *


LEFT_HOME_STATE = [-0.30, 0.20, 0.55, -2.20, 0.0, 2.55, 0.785398]
RIGHT_HOME_STATE = [0.30, 0.20, -0.55, -2.20, 0.0, 2.55, 0.785398]

TASK_ID = "cafe_table_reset_coasters_v8"
WOOD_TABLE_COLOR = (0.42, 0.23, 0.10)
UPRIGHT_CUP_QUAT = (0.5, 0.5, 0.5, 0.5)
TIPPED_CUP_QUAT = (0.0, 2**-0.5, 0.0, 2**-0.5)
COFFEE_MACHINE_QUAT = (2**-0.5, 2**-0.5, 0.0, 0.0)
WASTE_BOX_CAMERA_QUAT = (2**-0.5, 2**-0.5, 0.0, 0.0)
HANDLELESS_CUP_SCALE_MULTIPLIER = 0.70
DIRTY_CUP_SCALE_MULTIPLIER = 0.8
WASTE_BOX_SCALE_MULTIPLIER = 0.65

WASTE_BOX_CENTER_XY = np.asarray([-0.30, 0.16])
WASTE_BOX_COLLISION_BOXES = (
    ((0.0005, 0.0040, -0.0036), (0.1000, 0.0040, 0.1000)),
    ((-0.0955, 0.0445, -0.0036), (0.0040, 0.0445, 0.1000)),
    ((0.0965, 0.0445, -0.0036), (0.0040, 0.0445, 0.1000)),
    ((0.0005, 0.0445, -0.0996), (0.0920, 0.0445, 0.0040)),
    ((0.0005, 0.0445, 0.0924), (0.0920, 0.0445, 0.0040)),
)
COASTER_CENTER_XY = np.asarray([-0.04, 0.23])
COASTER_INTERIOR_HALF_XY = np.asarray([0.055, 0.055])

MAX_LINEAR_SPEED_M_S = 0.05
MAX_ANGULAR_SPEED_RAD_S = 0.5
MAX_UPRIGHT_ANGLE_DEG = 20.0
HOME_TOLERANCE_RAD = 0.05
STABLE_SUCCESS_STEPS = 250


class cafe_table_reset(Base_Task):
    """Keep one staged handleless cup on its coaster."""

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
            convex=True,
            scale_multiplier=1.0,
            collision_boxes=None,
        ):
            result = create_actor(
                scene=self,
                pose=sapien.Pose(position, quat),
                modelname=modelname,
                convex=convex,
                model_id=model_id,
                is_static=is_static,
                scale_multiplier=scale_multiplier,
                collision_boxes=collision_boxes,
            )
            result.set_name(instance_name)
            if not is_static:
                result.set_mass(mass)
            return result

        self.used_cups = [
            actor(
                "021_cup",
                [COASTER_CENTER_XY[0], COASTER_CENTER_XY[1], 0.749],
                instance_name="handleless_clean_cup",
                model_id=4,
                scale_multiplier=HANDLELESS_CUP_SCALE_MULTIPLIER,
            ),
            actor(
                "901_dirty_coffee_cup",
                [0.14, -0.10, 0.79],
                instance_name="tipped_used_cup",
                model_id=0,
                quat=TIPPED_CUP_QUAT,
                scale_multiplier=DIRTY_CUP_SCALE_MULTIPLIER,
            ),
        ]
        self.waste_basket = actor(
            "902_dirty_waste_box",
            [WASTE_BOX_CENTER_XY[0], WASTE_BOX_CENTER_XY[1], 0.741],
            instance_name="waste_basket",
            model_id=0,
            is_static=True,
            convex=False,
            quat=WASTE_BOX_CAMERA_QUAT,
            scale_multiplier=WASTE_BOX_SCALE_MULTIPLIER,
            collision_boxes=WASTE_BOX_COLLISION_BOXES,
        )
        self.coasters = [
            actor(
                "019_coaster",
                [COASTER_CENTER_XY[0], COASTER_CENTER_XY[1], 0.741],
                instance_name="wall_coaster",
                model_id=0,
                is_static=True,
            )
        ]
        self.coffee_machine = actor(
            "900_coffee_machine",
            [0.20, 0.16, 0.741],
            instance_name="coffee_machine",
            model_id=0,
            is_static=True,
            convex=False,
            quat=COFFEE_MACHINE_QUAT,
        )
        self._stable_success_steps = 0
        self.subgoal_vector = [False]

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

        position = np.asarray(self.coasters[0].get_pose().p, dtype=float).copy()
        position[2] += 0.008
        self._transfer(
            self.used_cups[0],
            "left",
            sapien.Pose(position, UPRIGHT_CUP_QUAT),
        )

        self.info["info"] = {
            "{handleless_clean_cup}": "021_cup/base4",
            "{tipped_used_cup}": "901_dirty_coffee_cup/base0",
            "{wall_coaster}": "019_coaster/base0",
            "{waste_basket}": "902_dirty_waste_box/base0",
            "{coffee_machine}": "900_coffee_machine/base0",
            "task_id": TASK_ID,
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

    @staticmethod
    def _upright(actor):
        rotation = np.asarray(
            actor.get_pose().to_transformation_matrix()[:3, :3],
            dtype=float,
        )
        world_up_alignment = float(
            rotation[:, 1] @ np.asarray([0.0, 0.0, 1.0])
        )
        return world_up_alignment >= np.cos(np.deg2rad(MAX_UPRIGHT_ANGLE_DEG))

    def _cup_on_coaster(self, cup, coaster):
        return bool(
            self._inside(
                cup,
                np.asarray(coaster.get_pose().p[:2], dtype=float),
                COASTER_INTERIOR_HALF_XY,
            )
            and self._supported(cup, coaster.get_name())
            and self._upright(cup)
            and self._stable(cup)
            and self._unheld(cup)
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
        coaster_goal = self._cup_on_coaster(
            self.used_cups[0],
            self.coasters[0],
        )
        self.subgoal_vector = [coaster_goal]
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
