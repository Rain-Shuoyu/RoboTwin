"""Nine RoboDojo rigid tasks sharing the existing RoboTwin robot and action path.

Task/layout source: RoboDojo f57e1633ffed59761bc8e568cfe7530c4b33381c.
Asset revision: 269457a3b42d8d0ae69898a601f2c1487c2f826e.
This SAPIEN port is a separate evaluation setting, not a reproduction of the
original Isaac Sim leaderboard. See env_cfg/robodojo/README.md.
"""
from __future__ import annotations

import json
import operator
import os
from pathlib import Path

import numpy as np
import sapien

from ._base_task import Base_Task
from .utils.create_actor import create_box, create_table
from .utils.robodojo_assets import load_converted_asset
from .utils.robodojo_predicates import TaskEvaluator, load_task_definitions


class robodojo_rigid(Base_Task):
    def setup_demo(self, is_test=False, **kwargs):
        parameters = kwargs.pop("task_parameters")
        self.source_task = parameters["source_task"]
        definition = load_task_definitions()["tasks"][self.source_task]
        self.source_seed = operator.index(parameters.get("source_seed", 0))
        self.layout_index = operator.index(kwargs.get("seed", 0))
        self.asset_root = Path(os.environ.get(
            "ROBOTWIN_ROBODOJO_ASSET_ROOT",
            str(Path(__file__).resolve().parents[1] / "assets/objects/robodojo_rigid"),
        )).expanduser()
        self.layout_path = self.asset_root / "layouts" / str(self.source_seed) / f"{self.source_task}_{self.layout_index}.json"
        self.source_layout = json.loads(self.layout_path.read_text())
        self.robodojo_assets = {}
        self.robodojo_category_indices = {}
        super()._init_task_env_(**kwargs)
        self.task_evaluator = TaskEvaluator(
            self.source_task, self.robodojo_assets, self._robot_poses,
            category_indices=self.robodojo_category_indices,
            height_shift=self.robodojo_height_shift,
        )
        self.step_lim = definition["step_limit"]

    def create_table_and_wall(self, table_xy_bias=(0, 0), table_height=.74,
                              table_texture_override=None, table_color_override=None):
        """Preserve source tabletop footprint using RoboTwin's native primitives.

        The source surface is z=.765 m. Every task and its absolute Z predicate
        receives the same translation to the current native surface at .74 m
        (plus the existing native table-height bias, if explicitly configured).
        Robot bases, cameras, simulation ownership, and action loop are inherited.
        """
        source_table = self.source_layout["Table"]
        length, width, thickness = source_table["scale"]
        center = source_table["default_pos"]
        self.table_xy_bias = list(center[:2])
        top = table_height + self.table_z_bias
        self.robodojo_height_shift = top - .765
        self.wall_texture = self.table_texture = None
        if self.random_background:
            texture_type = "unseen" if self.eval_mode else "seen"
            directory = Path("assets/background_texture") / texture_type
            file_count = len([path for path in directory.iterdir() if path.is_file()])
            wall_texture, table_texture = np.random.randint(0, file_count), np.random.randint(0, file_count)
            self.wall_texture = f"{texture_type}/{wall_texture}"
            self.table_texture = f"{texture_type}/{table_texture}"
            if np.random.rand() <= self.clean_background_rate:
                self.wall_texture = self.table_texture = None
        if table_texture_override is not None:
            self.table_texture = table_texture_override
        if table_color_override is not None:
            self.table_texture = None
        self.wall = create_box(
            self.scene, sapien.Pose(p=[0, center[1] + width / 2 + .6, 1.5]), half_size=[3, .6, 1.5],
            color=(1, .9, .9), name="wall", texture_id=self.wall_texture, is_static=True,
        )
        self.table = create_table(
            self.scene, pose=sapien.Pose(p=[center[0], center[1], top]), length=length,
            width=width, height=top, thickness=thickness,
            color=(1, 1, 1) if table_color_override is None else table_color_override,
            is_static=True, texture_id=self.table_texture,
        )

    def load_actors(self):
        for kind in ("Rigid", "Geometry", "Articulation"):
            for outer_category, instances in self.source_layout.get(kind, {}).items():
                for ordinal, instance in enumerate(instances):
                    category = instance.get("category", outer_category)
                    if category == "camera_stand":
                        continue  # Replaced by the inherited native cameras.
                    source_kind = "Clutter" if instance.get("type") == "cluttered" else kind
                    index = operator.index(instance["category_idx"])
                    manifest_path = self.asset_root / "assets" / source_kind / category / f"{index:05d}" / "asset.json"
                    label = instance.get("label") or f"clutter/{source_kind}/{category}/{ordinal}"
                    position = np.asarray(instance["default_pos"], dtype=float).copy()
                    position[2] += self.robodojo_height_shift
                    pose = sapien.Pose(p=position, q=instance["default_ori"])
                    physics = instance.get("physics", {})
                    asset = load_converted_asset(
                        self.scene, manifest_path, pose, fixed_root=kind == "Geometry",
                        scale=instance["scale"], physics=physics,
                    )
                    self.robodojo_assets[label] = asset
                    self.robodojo_category_indices[label] = index

    def _robot_poses(self):
        return {"left": self.robot.get_left_ee_pose(), "right": self.robot.get_right_ee_pose()}

    def check_success(self):
        return self.task_evaluator.check_success()
