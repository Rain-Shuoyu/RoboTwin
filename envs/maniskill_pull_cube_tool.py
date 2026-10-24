# SPDX-License-Identifier: Apache-2.0
"""RoboTwin port of ManiSkill v3.0.1 PullCubeTool-v1.

Source: https://github.com/mani-skill/ManiSkill/blob/v3.0.1/mani_skill/envs/tasks/tabletop/pull_cube_tool.py
Copyright: ManiSkill contributors. License copy: envs/LICENSE.ManiSkill.
Modified for RoboTwin: native Base_Task lifecycle, NumPy episode sampling,
and a configurable rigid layout transform for the existing pair of robots.
The two tool boxes, source density/material, random ranges and <0.6 m
horizontal-distance success condition are preserved. No expert actions run.

Default layout: rotate +90 degrees, translate (-0.10, 0.26, 0.74) m; shift
the unchanged 1.2 x 0.7 m table and wall +0.20 m in y. The cube centers are
x=[-0.15,0.15], y=[0.31,0.51] m, beyond the common near working area of bases
at (+/-0.30,-0.65,0.75) m. This is a layout candidate: actual direct-grasp
exclusion, tool reach, collision-free motion and camera visibility have NOT
been verified. In particular the far end of the full source random range may
also be unreachable with the original 0.20 m tool; do not infer solvability
from horizontal distance or a successful reset.

Public task_parameters: source_to_world_translation (metres),
source_to_world_yaw (radians), table_xy_bias (metres), success_arm (left/right).
The native table-height bias is applied to the translation's z component.
These parameters place a scene; none are used to generate robot actions.
"""
from __future__ import annotations

import math

import numpy as np
import sapien

from ._base_task import Base_Task


class maniskill_pull_cube_tool(Base_Task):
    cube_half_size = 0.02
    handle_length = 0.2
    hook_length = 0.05
    width = 0.05
    height = 0.05
    cube_size = 0.02
    arm_reach = 0.35

    def setup_demo(self, **kwargs):
        parameters = kwargs.pop("task_parameters", {})
        self.layout_translation = np.asarray(
            parameters.get("source_to_world_translation", [-0.1, 0.26, 0.74]), dtype=float
        )
        self.layout_yaw = float(parameters.get("source_to_world_yaw", math.pi / 2))
        self.success_arm = parameters.get("success_arm", "left")
        if self.success_arm not in ("left", "right"):
            raise ValueError("success_arm must be left or right")
        kwargs.setdefault("table_xy_bias", parameters.get("table_xy_bias", [0, 0.2]))
        super()._init_task_env_(**kwargs)

    def create_table_and_wall(self, table_xy_bias=(0, 0.2), **kwargs):
        # Base_Task keeps its wall at y=1 regardless of table_xy_bias. Move the
        # wall with this task's table: its front is then y=.60, beyond the cube
        # random region and cube corners (maximum y approximately .538).
        super().create_table_and_wall(table_xy_bias=table_xy_bias, **kwargs)
        pose = self.wall.get_pose()
        position = np.asarray(pose.p).copy()
        position[:2] += table_xy_bias
        self.wall.set_pose(sapien.Pose(position, pose.q))

    def _build_l_shaped_tool(self, handle_length, hook_length, width, height):
        # This is the upstream construction, in its original local frame and
        # order. The hook's density remains SAPIEN's default, as in the source.
        builder = self.scene.create_actor_builder()
        mat = sapien.render.RenderMaterial()
        mat.set_base_color([1, 0, 0, 1])
        mat.metallic = 1.0
        mat.roughness = 0.0
        mat.specular = 1.0
        builder.add_box_collision(
            sapien.Pose([handle_length / 2, 0, 0]),
            [handle_length / 2, width / 2, height / 2],
            density=500,
        )
        builder.add_box_visual(
            sapien.Pose([handle_length / 2, 0, 0]),
            [handle_length / 2, width / 2, height / 2],
            material=mat,
        )
        builder.add_box_collision(
            sapien.Pose([handle_length - hook_length / 2, width, 0]),
            [hook_length / 2, width, height / 2],
        )
        builder.add_box_visual(
            sapien.Pose([handle_length - hook_length / 2, width, 0]),
            [hook_length / 2, width, height / 2],
            material=mat,
        )
        return builder.build(name="l_shape_tool")

    def load_actors(self):
        translation = self.layout_translation.copy()
        translation[2] += self.table_z_bias
        self.source_to_world = sapien.Pose(
            translation, [math.cos(self.layout_yaw / 2), 0, 0, math.sin(self.layout_yaw / 2)]
        )
        builder = self.scene.create_actor_builder()
        builder.add_box_collision(half_size=[self.cube_half_size] * 3)
        builder.add_box_visual(
            half_size=[self.cube_half_size] * 3,
            material=sapien.render.RenderMaterial(base_color=np.array([12, 42, 160, 255]) / 255),
        )
        self.cube = builder.build(name="cube")
        self.l_shape_tool = self._build_l_shaped_tool(
            self.handle_length, self.hook_length, self.width, self.height
        )

        tool_xyz = np.zeros(3)
        tool_xyz[:2] = -np.random.rand(2) * 0.2 - 0.1
        tool_xyz[2] = self.height / 2
        self.l_shape_tool.set_pose(self.source_to_world * sapien.Pose(tool_xyz))

        cube_xyz = np.zeros(3)
        cube_xyz[0] = self.arm_reach + np.random.rand() * self.handle_length - 0.3
        cube_xyz[1] = np.random.rand() * 0.3 - 0.25
        cube_xyz[2] = self.cube_size / 2 + 0.015
        yaw = np.random.uniform(-np.pi / 6, np.pi / 6)
        cube_q = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
        self.cube.set_pose(self.source_to_world * sapien.Pose(cube_xyz, cube_q))

    def check_success(self):
        # The source checks only xy distance to its single robot base. Select
        # one actual base in the pair rather than inventing a midpoint goal or
        # accepting proximity to either robot, which would change that rule.
        cube_pos = self.cube.get_pose().p
        base_pos = getattr(self.robot, f"{self.success_arm}_entity").get_root_pose().p
        return bool(np.linalg.norm(cube_pos[:2] - base_pos[:2]) < 0.6)
