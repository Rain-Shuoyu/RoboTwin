"""Compact cafe-table clearing and reset task for the standard RoboTwin table."""

import numpy as np
import sapien

from ._base_task import Base_Task
from .utils import *


class cafe_table_reset(Base_Task):
    """Clear a customer table, stage supplies, and signal completion."""

    def setup_demo(self, **kwargs):
        super()._init_task_env_(**kwargs, table_texture_override="custom/restaurant_dark_grid_10cm")

    def _zone_marker(self, position, half_size, color, name):
        return create_box(
            scene=self.scene,
            pose=sapien.Pose(p=position, q=[1, 0, 0, 0]),
            half_size=half_size,
            color=color,
            is_static=True,
            name=name,
        )

    def load_actors(self):
        # Low, colored tabletop markers make the three functional zones visible
        # in the same head-camera composition used by the hamburger task.
        self.customer_zone = self._zone_marker(
            [-0.28, -0.02, 0.742], [0.19, 0.16, 0.001], (0.72, 0.43, 0.25), "customer_table_zone"
        )
        self.collection_zone = self._zone_marker(
            [0.00, -0.02, 0.742], [0.16, 0.16, 0.001], (0.95, 0.78, 0.20), "collection_zone"
        )
        self.cleaning_zone = self._zone_marker(
            [0.31, -0.02, 0.742], [0.16, 0.16, 0.001], (0.32, 0.72, 0.42), "cleaning_supply_zone"
        )

        def actor(modelname, position, *, model_id=0, is_static=False, mass=0.05, quat=None):
            result = create_actor(
                scene=self,
                pose=sapien.Pose(position, quat or [0.707, 0.707, 0, 0]),
                modelname=modelname,
                convex=True,
                model_id=model_id,
                is_static=is_static,
            )
            if not is_static:
                result.set_mass(mass)
            return result

        self.cup = actor("021_cup", [-0.40, -0.10, 0.741])
        self.mug = actor("039_mug", [-0.25, -0.22, 0.741])
        self.plate = actor("003_plate", [-0.10, -0.11, 0.741])
        self.tissue_box = actor("023_tissue-box", [-0.38, 0.11, 0.741])
        self.tray = actor("008_tray", [0.00, -0.02, 0.741], mass=0.12, quat=[0.706527, 0.706483, -0.0291356, -0.0291767])
        self.table_bin = actor("063_tabletrashbin", [0.38, 0.16, 0.741], is_static=True, quat=[0.5, 0.5, 0.5, 0.5])
        self.cleaner = actor("096_cleaner", [0.22, 0.15, 0.741])
        self.roll_paper = actor("028_roll-paper", [-0.15, 0.15, 0.741])
        self.breadbasket = actor("076_breadbasket", [0.28, -0.16, 0.741], mass=0.12)
        self.bread = actor("075_bread", [-0.42, 0.17, 0.741], mass=0.04)
        self.bell = actor("050_bell", [0.48, -0.22, 0.741], is_static=True, quat=[0.5, 0.5, 0.5, 0.5])

        self.check_arm_function = self.is_right_gripper_close

    def _place_on_tray(self, item, arm_tag, offset):
        tray_fp = self.tray.get_functional_point(0)
        target = sapien.Pose(
            [tray_fp.p[0] + offset[0], tray_fp.p[1] + offset[1], tray_fp.p[2] + 0.025],
            [1, 0, 0, 0],
        )
        self.move(self.grasp_actor(item, arm_tag=arm_tag, pre_grasp_dis=0.08))
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.08, move_axis="arm"))
        self.move(self.place_actor(item, arm_tag=arm_tag, target_pose=target, constrain="free", pre_dis=0.08))

    def _place_in_region(self, item, arm_tag, target_xy, *, z=0.80):
        target = sapien.Pose([target_xy[0], target_xy[1], z], [1, 0, 0, 0])
        self.move(self.grasp_actor(item, arm_tag=arm_tag, pre_grasp_dis=0.08))
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.08, move_axis="arm"))
        self.move(self.place_actor(item, arm_tag=arm_tag, target_pose=target, constrain="free", pre_dis=0.08))

    def play_once(self):
        self._place_on_tray(self.cup, "left", [-0.06, 0.00])
        self._place_on_tray(self.mug, "right", [0.06, 0.00])
        self._place_on_tray(self.plate, "left", [0.00, 0.06])

        self._place_in_region(self.tray, "right", [0.31, -0.02], z=0.80)
        self._place_in_region(self.roll_paper, "left", [0.38, 0.16], z=0.82)
        self._place_in_region(self.tissue_box, "right", [0.22, 0.10], z=0.82)
        self._place_in_region(self.cleaner, "left", [0.36, 0.02], z=0.82)
        self._place_in_region(self.bread, "right", [0.28, -0.16], z=0.82)

        self.move(self.grasp_actor(self.bell, arm_tag="right", pre_grasp_dis=0.10, grasp_dis=0.10, contact_point_id=0))
        self.move(self.move_by_displacement("right", z=-0.045))
        self.stage_success_tag = self._bell_contact()
        self.move(self.move_by_displacement("right", z=0.045))
        self.move(self.open_gripper("right"))
        self.info["info"] = {"{A}": "021_cup/base0", "{B}": "039_mug/base0", "{C}": "003_plate/base0"}
        return self.info

    @staticmethod
    def _in_region(actor, center, tolerance=(0.10, 0.10)):
        position = actor.get_pose().p
        return bool(np.all(np.abs(position[:2] - np.asarray(center)) < np.asarray(tolerance)))

    def _bell_contact(self):
        bell_pose = self.bell.get_contact_point(0)[:3]
        for position in self.get_gripper_actor_contact_position("050_bell"):
            if np.all(np.abs(position[:2] - bell_pose[:2]) < [0.025, 0.025]) and abs(position[2] - bell_pose[2]) < 0.03:
                return True
        return False

    def check_success(self):
        tray_center = self.tray.get_pose().p[:2]
        cleaning_center = self.cleaning_zone.get_pose().p[:2]
        bin_center = self.table_bin.get_pose().p[:2]
        basket_center = self.breadbasket.get_pose().p[:2]
        supplies_ready = self._in_region(self.tissue_box, [0.22, 0.10]) and self._in_region(self.cleaner, [0.36, 0.02])
        dishes_cleared = all(self._in_region(item, tray_center, (0.14, 0.12)) for item in (self.cup, self.mug, self.plate))
        return bool(
            dishes_cleared
            and self._in_region(self.tray, cleaning_center, (0.14, 0.12))
            and self._in_region(self.roll_paper, bin_center, (0.10, 0.10))
            and self._in_region(self.bread, basket_center, (0.12, 0.12))
            and supplies_ready
            and self.stage_success_tag
            and self.robot.is_left_gripper_open()
            and self.robot.is_right_gripper_open()
        )
