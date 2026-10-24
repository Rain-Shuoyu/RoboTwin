# SPDX-License-Identifier: Apache-2.0
"""RoboTwin port of ManiSkill v3.0.1 PegInsertionSide-v1.

Source: https://github.com/mani-skill/ManiSkill/blob/v3.0.1/mani_skill/envs/tasks/tabletop/peg_insertion_side.py
Copyright: ManiSkill contributors. License copy: envs/LICENSE.ManiSkill.
Modified for RoboTwin: native Base_Task lifecycle, NumPy episode sampling and
a configurable rigid layout transform; no ManiSkill dependency or expert run.
The original single peg collision box, orange/white visual halves, four wall
boxes, randomized dimensions/poses, 3 mm clearance and hole-frame head-position
predicate are preserved. The source predicate has no orientation or upper-x
test; this port deliberately does not add one.
Object collisions explicitly use the source DefaultMaterialsConfig values
(static friction .3, dynamic friction .3, restitution 0), independently of
RoboTwin's scene-wide default and without changing table or robot materials.

Default layout: rotate +90 degrees and translate (0.10,-0.10,0.74) m onto
the native tabletop. This preserves every relative object pose and dimension;
robot reach, insertion physics and camera visibility remain unverified.
Public task_parameters: source_to_world_translation (metres),
source_to_world_yaw (radians), table_xy_bias (metres). The native table-height
bias is applied to the translation's z component. These parameters only place
the scene and never drive robot actions.
"""
from __future__ import annotations

import math

import numpy as np
import sapien

from ._base_task import Base_Task


def _build_box_with_hole(scene, inner_radius, outer_radius, depth, center=(0, 0), *, physical_material):
    # Upstream geometry, including asymmetric wall widths for the offset hole.
    # The local x axis is the insertion direction.
    builder = scene.create_actor_builder()
    thickness = (outer_radius - inner_radius) * 0.5
    half_center = [x * 0.5 for x in center]
    half_sizes = [
        [depth, thickness - half_center[0], outer_radius],
        [depth, thickness + half_center[0], outer_radius],
        [depth, outer_radius, thickness - half_center[1]],
        [depth, outer_radius, thickness + half_center[1]],
    ]
    offset = thickness + inner_radius
    poses = [
        sapien.Pose([0, offset + half_center[0], 0]),
        sapien.Pose([0, -offset + half_center[0], 0]),
        sapien.Pose([0, 0, offset + half_center[1]]),
        sapien.Pose([0, 0, -offset + half_center[1]]),
    ]
    mat = sapien.render.RenderMaterial(
        base_color=[255 / 255, 210 / 255, 137 / 255, 1], roughness=0.5, specular=0.5
    )
    for half_size, pose in zip(half_sizes, poses):
        builder.add_box_collision(pose, half_size, material=physical_material)
        builder.add_box_visual(pose, half_size, material=mat)
    return builder


class maniskill_peg_insertion_side(Base_Task):
    _clearance = 0.003

    def setup_demo(self, **kwargs):
        parameters = kwargs.pop("task_parameters", {})
        self.layout_translation = np.asarray(
            parameters.get("source_to_world_translation", [0.1, -0.1, 0.74]), dtype=float
        )
        self.layout_yaw = float(parameters.get("source_to_world_yaw", math.pi / 2))
        kwargs.setdefault("table_xy_bias", parameters.get("table_xy_bias", [0, 0]))
        super()._init_task_env_(**kwargs)

    def load_actors(self):
        translation = self.layout_translation.copy()
        translation[2] += self.table_z_bias
        self.source_to_world = sapien.Pose(
            translation, [math.cos(self.layout_yaw / 2), 0, 0, math.sin(self.layout_yaw / 2)]
        )
        # This compatibility API constructs a task-local PhysxMaterial;
        # assigning scene.default_physical_material would not set the engine
        # default used by add_box_collision when its material is omitted.
        self.object_physical_material = self.scene.create_physical_material(0.3, 0.3, 0)
        length = float(np.random.uniform(0.085, 0.125))
        radius = float(np.random.uniform(0.015, 0.025))
        center = 0.5 * (length - radius) * np.random.uniform(-1, 1, size=2)
        self.peg_half_size = np.array([length, radius, radius])
        self.peg_head_offset = sapien.Pose([length, 0, 0])
        self.box_hole_offset = sapien.Pose([0, *center])
        self.box_hole_radius = radius + self._clearance

        builder = self.scene.create_actor_builder()
        builder.add_box_collision(
            half_size=[length, radius, radius], material=self.object_physical_material
        )
        mat = sapien.render.RenderMaterial(
            base_color=[236 / 255, 115 / 255, 87 / 255, 1], roughness=0.5, specular=0.5
        )
        builder.add_box_visual(
            sapien.Pose([length / 2, 0, 0]),
            half_size=[length / 2, radius, radius], material=mat,
        )
        mat = sapien.render.RenderMaterial(
            base_color=[237 / 255, 246 / 255, 249 / 255, 1], roughness=0.5, specular=0.5
        )
        builder.add_box_visual(
            sapien.Pose([-length / 2, 0, 0]),
            half_size=[length / 2, radius, radius], material=mat,
        )
        builder.set_initial_pose(self.source_to_world * sapien.Pose([0, 0, 0.1]))
        self.peg = builder.build("peg")
        builder = _build_box_with_hole(
            self.scene, radius + self._clearance, length, length, center=center,
            physical_material=self.object_physical_material,
        )
        builder.set_initial_pose(self.source_to_world * sapien.Pose([0, 1, 0.1]))
        self.box = builder.build_kinematic("box_with_hole")

        peg_xy = np.random.uniform(low=[-0.1, -0.3], high=[0.1, 0])
        peg_yaw = np.random.uniform(np.pi / 2 - np.pi / 3, np.pi / 2 + np.pi / 3)
        self.peg.set_pose(self.source_to_world * sapien.Pose(
            [*peg_xy, radius], [math.cos(peg_yaw / 2), 0, 0, math.sin(peg_yaw / 2)]
        ))
        box_xy = np.random.uniform(low=[-0.05, 0.2], high=[0.05, 0.4])
        box_yaw = np.random.uniform(np.pi / 2 - np.pi / 8, np.pi / 2 + np.pi / 8)
        self.box.set_pose(self.source_to_world * sapien.Pose(
            [*box_xy, length], [math.cos(box_yaw / 2), 0, 0, math.sin(box_yaw / 2)]
        ))

    @property
    def peg_head_pose(self):
        return self.peg.get_pose() * self.peg_head_offset

    @property
    def box_hole_pose(self):
        return self.box.get_pose() * self.box_hole_offset

    def has_peg_inserted(self):
        peg_head_pos_at_hole = (self.box_hole_pose.inv() * self.peg_head_pose).p
        x_flag = -0.015 <= peg_head_pos_at_hole[0]
        y_flag = -self.box_hole_radius <= peg_head_pos_at_hole[1] <= self.box_hole_radius
        z_flag = -self.box_hole_radius <= peg_head_pos_at_hole[2] <= self.box_hole_radius
        return bool(x_flag and y_flag and z_flag), peg_head_pos_at_hole

    def check_success(self):
        return self.has_peg_inserted()[0]
