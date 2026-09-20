"""Preview-only garbage-sorting scene on RoboTwin's standard table."""

from dataclasses import dataclass

import numpy as np
import sapien

from ._base_task import Base_Task
from .utils import *


LEFT_HOME_STATE = [-0.30, 0.20, 0.55, -2.20, 0.0, 2.55, 0.785398]
RIGHT_HOME_STATE = [0.30, 0.20, -0.55, -2.20, 0.0, 2.55, 0.785398]
TASK_ID = "garbage_sorting_preview_v1"
WOOD_TABLE_COLOR = (0.42, 0.23, 0.10)
TABLE_TOP_Z = 0.74
BIN_THICKNESS = 0.01


@dataclass(frozen=True)
class BinSpec:
    name: str
    meaning: str
    center_xy: tuple[float, float]
    outer_size: tuple[float, float, float]
    color: tuple[float, float, float]


@dataclass(frozen=True)
class TrashSpec:
    instance_name: str
    model_name: str
    model_id: int
    category: str
    mass_kg: float
    base_quaternion_wxyz: tuple[float, float, float, float]


@dataclass(frozen=True)
class SpawnSpec:
    trash: TrashSpec
    position: tuple[float, float, float]
    quaternion_wxyz: tuple[float, float, float, float]


BIN_SPECS = (
    BinSpec(
        "source_bin",
        "unsorted",
        (0.0, -0.16),
        (0.50, 0.27, 0.11),
        (0.67, 0.58, 0.45),
    ),
    BinSpec(
        "recyclable_bin",
        "recyclable",
        (-0.39, 0.245),
        (0.23, 0.17, 0.11),
        (0.12, 0.55, 0.25),
    ),
    BinSpec(
        "other_bin",
        "other",
        (0.0, 0.245),
        (0.23, 0.17, 0.11),
        (0.38, 0.38, 0.38),
    ),
    BinSpec(
        "hazardous_bin",
        "hazardous",
        (0.39, 0.245),
        (0.23, 0.17, 0.11),
        (0.95, 0.40, 0.08),
    ),
)

ROTATE_LOCAL_Y_UP = (2**-0.5, 2**-0.5, 0.0, 0.0)
ROTATE_STANDARD_UP = (0.5, 0.5, 0.5, 0.5)
TRASH_INVENTORY = (
    TrashSpec(
        "light_bulb",
        "906_light_bulb",
        0,
        "hazardous",
        0.04,
        ROTATE_LOCAL_Y_UP,
    ),
    TrashSpec(
        "chemical_cleaner",
        "905_chemical_cleaner",
        0,
        "hazardous",
        0.08,
        ROTATE_LOCAL_Y_UP,
    ),
    TrashSpec(
        "bitten_apple",
        "907_bitten_apple",
        0,
        "other",
        0.05,
        ROTATE_LOCAL_Y_UP,
    ),
    TrashSpec(
        "milk_carton",
        "038_milk-box",
        1,
        "recyclable",
        0.03,
        ROTATE_STANDARD_UP,
    ),
    TrashSpec(
        "plastic_bottle",
        "114_bottle",
        4,
        "recyclable",
        0.03,
        ROTATE_LOCAL_Y_UP,
    ),
    TrashSpec(
        "drink_can",
        "071_can",
        1,
        "recyclable",
        0.03,
        ROTATE_STANDARD_UP,
    ),
)

SPAWN_SLOTS = (
    (-0.15, -0.22, 0.91),
    (0.00, -0.22, 0.91),
    (0.15, -0.22, 0.91),
    (-0.10, -0.11, 1.00),
    (0.10, -0.11, 1.00),
    (0.00, -0.16, 1.10),
)


def _quaternion_multiply(first, second):
    aw, ax, ay, az = first
    bw, bx, by, bz = second
    return np.asarray(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=float,
    )


def _euler_quaternion(roll, pitch, yaw):
    half_roll, half_pitch, half_yaw = roll / 2, pitch / 2, yaw / 2
    cr, sr = np.cos(half_roll), np.sin(half_roll)
    cp, sp = np.cos(half_pitch), np.sin(half_pitch)
    cy, sy = np.cos(half_yaw), np.sin(half_yaw)
    return np.asarray(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        dtype=float,
    )


def sample_spawn_specs(rng):
    order = tuple(int(index) for index in rng.permutation(len(TRASH_INVENTORY)))
    result = []
    for slot, inventory_index in zip(SPAWN_SLOTS, order):
        jitter = rng.uniform(
            (-0.015, -0.012, -0.008),
            (0.015, 0.012, 0.008),
        )
        roll, pitch = rng.uniform(-np.pi / 5, np.pi / 5, size=2)
        yaw = float(rng.uniform(-np.pi, np.pi))
        delta = _euler_quaternion(float(roll), float(pitch), yaw)
        quaternion = _quaternion_multiply(
            delta,
            TRASH_INVENTORY[inventory_index].base_quaternion_wxyz,
        )
        quaternion /= np.linalg.norm(quaternion)
        position = np.asarray(slot, dtype=float) + jitter
        result.append(
            SpawnSpec(
                trash=TRASH_INVENTORY[inventory_index],
                position=tuple(float(value) for value in position),
                quaternion_wxyz=tuple(float(value) for value in quaternion),
            )
        )
    return tuple(result)


def create_open_bin(scene, spec):
    x, y = spec.center_xy
    width, depth, height = spec.outer_size
    floor_z = TABLE_TOP_Z + BIN_THICKNESS / 2
    wall_z = TABLE_TOP_Z + height / 2
    parts = (
        (
            "floor",
            (x, y, floor_z),
            (width / 2, depth / 2, BIN_THICKNESS / 2),
        ),
        (
            "front_wall",
            (x, y - depth / 2 + BIN_THICKNESS / 2, wall_z),
            (width / 2, BIN_THICKNESS / 2, height / 2),
        ),
        (
            "rear_wall",
            (x, y + depth / 2 - BIN_THICKNESS / 2, wall_z),
            (width / 2, BIN_THICKNESS / 2, height / 2),
        ),
        (
            "left_wall",
            (x - width / 2 + BIN_THICKNESS / 2, y, wall_z),
            (BIN_THICKNESS / 2, depth / 2, height / 2),
        ),
        (
            "right_wall",
            (x + width / 2 - BIN_THICKNESS / 2, y, wall_z),
            (BIN_THICKNESS / 2, depth / 2, height / 2),
        ),
    )
    return tuple(
        create_box(
            scene=scene,
            pose=sapien.Pose(position),
            half_size=half_size,
            color=spec.color,
            is_static=True,
            name=f"{spec.name}_{part_name}",
        )
        for part_name, position, half_size in parts
    )


class garbage_sorting(Base_Task):
    def set_episode_rng(self, seed):
        self.seed = seed
        self.episode_rng = np.random.RandomState(seed)

    def sample_spawn_specs(self):
        return sample_spawn_specs(np.random.RandomState(self.seed))

    def setup_demo(self, now_ep_num=0, seed=0, is_test=False, **kwargs):
        self.ep_num = now_ep_num
        self.is_test = is_test
        self.set_episode_rng(seed)

        kwargs = kwargs.copy()
        left_config = kwargs["left_embodiment_config"].copy()
        right_config = kwargs["right_embodiment_config"].copy()
        left_config["homestate"] = [LEFT_HOME_STATE.copy(), RIGHT_HOME_STATE.copy()]
        right_config["homestate"] = [LEFT_HOME_STATE.copy(), RIGHT_HOME_STATE.copy()]
        kwargs["left_embodiment_config"] = left_config
        kwargs["right_embodiment_config"] = right_config
        kwargs["table_color_override"] = WOOD_TABLE_COLOR
        super()._init_task_env_(
            now_ep_num=now_ep_num,
            seed=seed,
            is_test=is_test,
            **kwargs,
        )

    def load_actors(self):
        self.bin_components = {
            spec.name: create_open_bin(self, spec) for spec in BIN_SPECS
        }
        self.trash_objects = {}
        self.trash_categories = {}
        self.object_metadata = {}
        for spawn in sample_spawn_specs(self.episode_rng):
            item = spawn.trash
            actor = create_actor(
                scene=self,
                pose=sapien.Pose(spawn.position, spawn.quaternion_wxyz),
                modelname=item.model_name,
                model_id=item.model_id,
                convex=True,
            )
            if actor is None:
                raise FileNotFoundError(
                    f"missing RoboTwin asset {item.model_name}/base{item.model_id}"
                )
            actor.set_name(item.instance_name)
            actor.set_mass(item.mass_kg)
            self.trash_objects[item.instance_name] = actor
            self.trash_categories[item.instance_name] = item.category
            self.object_metadata[item.instance_name] = {
                "asset": f"{item.model_name}/base{item.model_id}",
                "category": item.category,
                "mass_kg": item.mass_kg,
            }
        self.trash_actors = self.trash_objects

    def play_once(self):
        object_metadata = {
            item.instance_name: {
                "asset": f"{item.model_name}/base{item.model_id}",
                "category": item.category,
                "mass_kg": item.mass_kg,
            }
            for item in TRASH_INVENTORY
        }
        self.object_metadata = object_metadata
        self.info["info"] = {
            **{
                f"{{{name}}}": metadata["asset"]
                for name, metadata in object_metadata.items()
            },
            "task_id": TASK_ID,
            "preview_only": True,
            "objects": object_metadata,
        }
        return self.info

    def check_success(self):
        return False
