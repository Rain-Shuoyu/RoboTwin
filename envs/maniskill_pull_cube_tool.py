# SPDX-License-Identifier: Apache-2.0
"""RoboTwin port of ManiSkill v3.0.1 PullCubeTool-v1.

Source: https://github.com/mani-skill/ManiSkill/blob/v3.0.1/mani_skill/envs/tasks/tabletop/pull_cube_tool.py
Copyright: ManiSkill contributors. License copy: envs/LICENSE.ManiSkill.
Modified for RoboTwin: native Base_Task lifecycle, NumPy episode sampling,
and a configurable rigid layout transform for the existing pair of robots.
The two tool boxes, source density/material, random ranges and <0.6 m
horizontal-distance success condition are preserved. No expert actions run.

Default layout: rotate +90 degrees and translate (-0.30,-0.035,0.74) m,
preserving the horizontal object positions relative to the original Panda
base (-0.615,0,0), placed by ManiSkill's v3.0.1 TableSceneBuilder.initialize.
The mapped base xy equals native left Franka's (-0.30,-0.65); the source
tabletop z=0 maps to native z=.74, while the native base is at z=.75. This
retains a 1 cm base-to-table height difference from the source environment.
Cube centers are x=[-0.35,-0.05], y=[0.015,0.215] m. The native table and wall
keep their standard placement. This task uses the left arm; the other arm
does not participate. That is a task instruction, not a physical restriction
or a claim that both arms cannot directly reach the cube. Tool reach,
collision-free motion and camera visibility remain unverified; a successful
reset does not prove task solvability.

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
            parameters.get("source_to_world_translation", [-0.3, -0.035, 0.74]), dtype=float
        )
        self.layout_yaw = float(parameters.get("source_to_world_yaw", math.pi / 2))
        self.success_arm = parameters.get("success_arm", "left")
        if self.success_arm not in ("left", "right"):
            raise ValueError("success_arm must be left or right")
        kwargs.setdefault("table_xy_bias", parameters.get("table_xy_bias", [0, 0]))
        super()._init_task_env_(**kwargs)

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
