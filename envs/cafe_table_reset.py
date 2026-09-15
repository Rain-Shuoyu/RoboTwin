"""Two-guest cafe turnover task on RoboTwin's standard table."""

import numpy as np
import sapien

from ._base_task import Base_Task
from .utils import *


LEFT_HOME_STATE = [-0.30, 0.20, 0.55, -2.20, 0.0, 2.55, 0.785398]
RIGHT_HOME_STATE = [0.30, 0.20, -0.55, -2.20, 0.0, 2.55, 0.785398]

TRAY_CENTER_XY = np.asarray([-0.27, 0.10])
TRAY_INTERIOR_HALF_XY = np.asarray([0.12, 0.065])
SEAT_CENTERS_XY = (np.asarray([-0.23, -0.17]), np.asarray([0.23, -0.17]))
SEAT_INTERIOR_HALF_XY = np.asarray([0.055, 0.055])
BASKET_CENTER_XY = np.asarray([0.00, -0.15])
BASKET_INTERIOR_HALF_XY = np.asarray([0.075, 0.045])

MAX_LINEAR_SPEED_M_S = 0.05
MAX_ANGULAR_SPEED_RAD_S = 0.5
MAX_MUG_TILT_DEG = 20.0
HOME_TOLERANCE_RAD = 0.05
STABLE_SUCCESS_STEPS = 250


class cafe_table_reset(Base_Task):
    """Recover two old cups and replenish two places and the bread basket."""

    def setup_demo(self, **kwargs):
        kwargs = kwargs.copy()
        left_config = kwargs["left_embodiment_config"].copy()
        right_config = kwargs["right_embodiment_config"].copy()
        left_config["homestate"] = [LEFT_HOME_STATE.copy(), RIGHT_HOME_STATE.copy()]
        right_config["homestate"] = [LEFT_HOME_STATE.copy(), RIGHT_HOME_STATE.copy()]
        kwargs["left_embodiment_config"] = left_config
        kwargs["right_embodiment_config"] = right_config
        kwargs["table_texture_override"] = "custom/restaurant_dark_grid_10cm"
        super()._init_task_env_(**kwargs)

    def _zone_marker(self, position, half_size, color, name):
        return create_visual_box(
            scene=self.scene,
            pose=sapien.Pose(p=position, q=[1, 0, 0, 0]),
            half_size=half_size,
            color=color,
            name=name,
        )

    def load_actors(self):
        self.seat_markers = [
            self._zone_marker(
                [-0.23, -0.17, 0.742],
                [0.07, 0.07, 0.001],
                (0.72, 0.43, 0.25),
                "dining_mat_a",
            ),
            self._zone_marker(
                [0.23, -0.17, 0.742],
                [0.07, 0.07, 0.001],
                (0.72, 0.43, 0.25),
                "dining_mat_b",
            ),
        ]
        self.collection_marker = self._zone_marker(
            [-0.27, 0.10, 0.742],
            [0.17, 0.115, 0.001],
            (0.95, 0.78, 0.20),
            "recovery_zone",
        )
        self.supply_marker = self._zone_marker(
            [0.27, 0.10, 0.742],
            [0.20, 0.16, 0.001],
            (0.32, 0.72, 0.42),
            "replacement_supply_zone",
        )

        def actor(
            modelname,
            position,
            *,
            instance_name,
            model_id,
            is_static=False,
            mass=0.05,
            quat=None,
        ):
            result = create_actor(
                scene=self,
                pose=sapien.Pose(position, quat or [0.5, 0.5, 0.5, 0.5]),
                modelname=modelname,
                convex=True,
                model_id=model_id,
                is_static=is_static,
            )
            result.set_name(instance_name)
            if not is_static:
                result.set_mass(mass)
            return result

        self.old_cups = [
            actor(
                "021_cup",
                [-0.23, -0.17, 0.741],
                instance_name="old_cup_a",
                model_id=6,
            ),
            actor(
                "021_cup",
                [0.23, -0.17, 0.741],
                instance_name="old_cup_b",
                model_id=6,
            ),
        ]
        self.replacement_mugs = [
            actor(
                "039_mug",
                [0.17, 0.17, 0.741],
                instance_name="replacement_mug_1",
                model_id=3,
            ),
            actor(
                "039_mug",
                [0.36, 0.17, 0.741],
                instance_name="replacement_mug_2",
                model_id=3,
            ),
        ]
        self.breads = [
            actor(
                "075_bread",
                [0.20, 0.02, 0.741],
                instance_name="bread_1",
                model_id=1,
                mass=0.04,
            ),
            actor(
                "075_bread",
                [0.34, 0.02, 0.741],
                instance_name="bread_2",
                model_id=1,
                mass=0.04,
            ),
        ]
        self.tray = actor(
            "008_tray",
            [-0.27, 0.10, 0.741],
            instance_name="recovery_tray",
            model_id=0,
            is_static=True,
            quat=[0.706527, 0.706483, -0.0291356, -0.0291767],
        )
        self.breadbasket = actor(
            "076_breadbasket",
            [0.00, -0.15, 0.741],
            instance_name="bread_basket",
            model_id=0,
            is_static=True,
        )
        self.tissue_box = actor(
            "023_tissue-box",
            [0.00, 0.25, 0.741],
            instance_name="tissue_box",
            model_id=0,
            is_static=True,
        )

        self._old_cup_recovered = [False, False]
        self._seat_blocked_by_early_replacement = [False, False]
        self._stable_success_steps = 0
        self.subgoal_vector = [False] * 6

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

        tray_z = float(self.tray.get_pose().p[2] + 0.055)
        self._transfer(
            self.old_cups[0],
            "left",
            sapien.Pose([-0.325, 0.10, tray_z], [0.5, 0.5, 0.5, 0.5]),
        )
        self._transfer(
            self.old_cups[1],
            "right",
            sapien.Pose([-0.215, 0.10, tray_z], [0.5, 0.5, 0.5, 0.5]),
        )

        for mug, arm_tag, center in zip(
            self.replacement_mugs,
            ("left", "right"),
            SEAT_CENTERS_XY,
        ):
            self._transfer(
                mug,
                arm_tag,
                sapien.Pose([center[0], center[1], 0.741], [0.5, 0.5, 0.5, 0.5]),
            )

        basket_z = float(self.breadbasket.get_pose().p[2] + 0.055)
        for bread, target_x in zip(self.breads, (-0.04, 0.04)):
            self._transfer(
                bread,
                "right",
                sapien.Pose([target_x, -0.15, basket_z], [0.5, 0.5, 0.5, 0.5]),
            )

        self.info["info"] = {
            "{old_cup_a}": "021_cup/base6",
            "{old_cup_b}": "021_cup/base6",
            "{replacement_mug_1}": "039_mug/base3",
            "{replacement_mug_2}": "039_mug/base3",
            "{bread_1}": "075_bread/base1",
            "{bread_2}": "075_bread/base1",
            "task_id": "cafe_turnover_two_guests_v1",
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

    def _old_cup_goal(self, cup):
        return bool(
            self._inside(cup, TRAY_CENTER_XY, TRAY_INTERIOR_HALF_XY)
            and self._supported(cup, self.tray.get_name())
            and self._stable(cup)
            and self._unheld(cup)
        )

    def _mug_at_seat(self, mug, seat_center):
        rotation = np.asarray(
            mug.get_pose().to_transformation_matrix()[:3, :3], dtype=float
        )
        local_up_world = rotation[:, 1]
        upright = bool(
            np.dot(local_up_world, np.asarray([0.0, 0.0, 1.0]))
            >= np.cos(np.deg2rad(MAX_MUG_TILT_DEG))
        )
        return bool(
            self._inside(mug, seat_center, SEAT_INTERIOR_HALF_XY)
            and self._supported(mug, "table")
            and upright
            and self._stable(mug)
            and self._unheld(mug)
        )

    def _bread_goal(self, bread):
        return bool(
            self._inside(bread, BASKET_CENTER_XY, BASKET_INTERIOR_HALF_XY)
            and self._supported(bread, self.breadbasket.get_name())
            and self._stable(bread)
            and self._unheld(bread)
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
        previous_recovery = tuple(self._old_cup_recovered)
        old_goals = [self._old_cup_goal(cup) for cup in self.old_cups]

        mug_seat_matrix = [
            [self._mug_at_seat(mug, center) for mug in self.replacement_mugs]
            for center in SEAT_CENTERS_XY
        ]
        for seat_index, occupancy in enumerate(mug_seat_matrix):
            occupied = any(occupancy)
            if occupied and not previous_recovery[seat_index]:
                self._seat_blocked_by_early_replacement[seat_index] = True
            elif not occupied:
                self._seat_blocked_by_early_replacement[seat_index] = False

        seat_goals = [
            bool(
                previous_recovery[index]
                and not self._seat_blocked_by_early_replacement[index]
                and sum(mug_seat_matrix[index]) == 1
            )
            for index in range(2)
        ]
        mugs_used_once = all(
            sum(mug_seat_matrix[seat][mug] for seat in range(2)) == 1
            for mug in range(2)
        )
        if not mugs_used_once:
            seat_goals = [False, False]

        self._old_cup_recovered = [
            previous_recovery[index] or old_goals[index] for index in range(2)
        ]
        bread_goals = [self._bread_goal(bread) for bread in self.breads]
        self.subgoal_vector = old_goals + seat_goals + bread_goals

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
