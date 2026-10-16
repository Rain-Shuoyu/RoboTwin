"""Restaurant equipment layout candidate without a task-success evaluator."""
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import sapien

from .restaurant_pass_counter import restaurant_pass_counter, _configure_loose_prop
from .utils import create_actor


DEFAULT_LAYOUT = Path(__file__).resolve().parents[1] / "env_cfg/restaurant_advance.json"
BACKGROUND_TEXTURE_ROOT = Path(__file__).resolve().parents[1] / "assets/background_texture"


def _load_layout(layout_path):
    layout = json.loads(layout_path.read_text())
    if layout["schema_version"] != "restaurant_advance_layout_v1":
        raise ValueError("unsupported restaurant layout")
    lights = layout.setdefault("foundation_point_lights", [])
    for relative_path in layout.get("geometry_components", []):
        component_path = layout_path.parent / relative_path
        component = json.loads(component_path.read_text())
        if component["schema_version"] != "robotwin_fixed_geometry_v1":
            raise ValueError(f"unsupported fixed geometry component: {component_path}")
        layout["geometry"].extend(component["geometry"])
        lights.extend(component["foundation_point_lights"])
    return layout


class restaurant_advance(restaurant_pass_counter):
    def setup_demo(self, **kwargs):
        self.layout = _load_layout(DEFAULT_LAYOUT)
        kwargs = kwargs.copy()
        translation = self.layout["robot_base_translation_m"]
        # Apply the scene offset before inherited setup creates robots and planners.
        for side in ("left", "right"):
            key = f"{side}_embodiment_config"
            config = deepcopy(kwargs[key])
            config["robot_pose"] = [
                [pose[axis] + translation[axis] for axis in range(3)] + list(pose[3:])
                for pose in config["robot_pose"]
            ]
            kwargs[key] = config
        return super().setup_demo(**kwargs)

    def setup_scene(self, **kwargs):
        lighting = deepcopy(self.layout["lighting"])
        lighting.update(kwargs)
        return super().setup_scene(**lighting)

    def create_table_and_wall(self, table_xy_bias=(0, 0), table_height=0.74,
                              table_texture_override=None, table_color_override=None):
        # The inherited initialization still supplies the original robot homes.
        # This scene owns its fixed materials and does not change any cameras.
        self.table_xy_bias = list(table_xy_bias)
        self.wall_texture = self.table_texture = None
        self._build_foundation(table_xy_bias, table_height)

    def _build_foundation(self, table_xy_bias, table_height):
        materials = {}
        for name, descriptor in self.layout["materials"].items():
            material = sapien.render.RenderMaterial(base_color=descriptor["base_color"])
            material.roughness = descriptor["roughness"]
            material.metallic = descriptor["metallic"]
            if "texture" in descriptor:
                path = BACKGROUND_TEXTURE_ROOT / f'{descriptor["texture"]}.png'
                if not path.is_file():
                    raise FileNotFoundError(path)
                material.set_base_color_texture(sapien.render.RenderTexture2D(str(path)))
            materials[name] = material

        z_bias = table_height - self.layout["counter"]["top_z"] + self.table_z_bias
        self.foundation_actors = {}
        for descriptor in self.layout["geometry"]:
            builder = self.scene.create_actor_builder()
            builder.set_physx_body_type("static")
            boxes = (descriptor["boxes"] if "boxes" in descriptor
                     else [{"half_size": descriptor["half_size"]}])
            for part in boxes:
                local_pose = sapien.Pose(part.get("center", [0, 0, 0]),
                                         part.get("quaternion_wxyz", [1, 0, 0, 0]))
                if descriptor["collision"]:
                    builder.add_box_collision(
                        pose=local_pose, half_size=part["half_size"],
                        material=self.scene.default_physical_material,
                    )
                builder.add_box_visual(
                    pose=local_pose, half_size=part["half_size"],
                    material=materials[part.get("material", descriptor["material"])],
                )
            x, y, z = descriptor["center"]
            builder.set_initial_pose(sapien.Pose(
                [x + table_xy_bias[0], y + table_xy_bias[1], z + z_bias],
                descriptor.get("quaternion_wxyz", [1, 0, 0, 0]),
            ))
            name = descriptor["name"]
            actor = builder.build(name=name)
            if "front_texture" in descriptor:
                self._attach_front_texture(actor, descriptor["half_size"],
                                           materials[descriptor["front_texture"]])
            self.foundation_actors[name] = actor
        self.table = self.foundation_actors["table"]
        self.wall = self.foundation_actors["wall"]
        self.foundation_lights = []
        for light in self.layout.get("foundation_point_lights", []):
            x, y, z = light["position"]
            self.foundation_lights.append(self.scene.add_point_light(
                [x + table_xy_bias[0], y + table_xy_bias[1], z + z_bias],
                light["color"], shadow=light["shadow"],
            ))

    def _attach_front_texture(self, actor, half_size, material):
        # Use an explicit front quad instead of relying on a box's built-in UVs.
        # Front is -Y; the board rotation tips this normal upward toward the head view.
        hx, hy, hz = half_size
        hx, hz = hx - 0.008, hz - 0.008
        front_y = -hy - 0.0005
        vertices = np.asarray([[-hx, front_y, -hz], [hx, front_y, -hz],
                               [hx, front_y, hz], [-hx, front_y, hz]], dtype=np.float32)
        triangles = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)
        normals = np.asarray([[0, -1, 0]] * 4, dtype=np.float32)
        uv = np.asarray([[0, 1], [1, 1], [1, 0], [0, 0]], dtype=np.float32)
        shape = sapien.render.RenderShapeTriangleMesh(vertices, triangles, normals, uv, material)
        body = sapien.render.RenderBodyComponent()
        body.attach(shape)
        actor.add_component(body)

    def load_actors(self):
        self._load_equipment()
        for descriptor in self.layout["props"]:
            if "placement" in descriptor:
                placement = descriptor["placement"]
                pose = self.equipment_poses[placement["equipment"]] * sapien.Pose(
                    placement["position_local"], descriptor["quaternion_wxyz"],
                )
            else:
                pose = sapien.Pose(descriptor["position"], descriptor["quaternion_wxyz"])
            actor = create_actor(
                scene=self, pose=pose,
                modelname=descriptor["modelname"], model_id=descriptor["model_id"],
                is_static=descriptor["is_static"], convex=descriptor["convex"],
            )
            if descriptor["loose_prop"]:
                _configure_loose_prop(actor)
            setattr(self, descriptor["attribute"], actor)

    def _load_equipment(self):
        self.equipment_actors = {}
        self.equipment_poses = {}
        self.equipment_lights = []
        for descriptor in self.layout["equipment"]:
            if not descriptor["enabled"]:
                continue
            mount = self.layout["equipment_mounts"][descriptor["mount"]]
            pose = sapien.Pose([*mount["center_xy"], mount["support_z"]],
                               descriptor["quaternion_wxyz"])
            actor = create_actor(
                scene=self, pose=pose,
                modelname=descriptor["modelname"], model_id=descriptor["model_id"],
                is_static=descriptor["is_static"], convex=descriptor["convex"],
            )
            self.equipment_actors[descriptor["attribute"]] = actor
            self.equipment_poses[descriptor["attribute"]] = pose
            setattr(self, descriptor["attribute"], actor)
            for light in descriptor.get("point_lights", []):
                light_pose = pose * sapien.Pose(light["position_local"])
                position = list(light_pose.p)
                position[2] += self.table_z_bias
                self.equipment_lights.append(self.scene.add_point_light(
                    position, light["color"], shadow=light["shadow"],
                ))

    def check_success(self):
        return False
