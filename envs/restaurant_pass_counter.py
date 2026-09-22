from ._base_task import Base_Task
from .utils import *
import sapien


LEFT_HOME_STATE = [-0.30, 0.20, 0.55, -2.20, 0.0, 2.55, 0.785398]
RIGHT_HOME_STATE = [0.30, 0.20, -0.55, -2.20, 0.0, 2.55, 0.785398]
LOOSE_PROP_MASS_KG = 0.05
LOOSE_PROP_MAX_DEPENETRATION_VELOCITY_M_S = 0.5


def _configure_loose_prop(actor):
    actor.set_mass(LOOSE_PROP_MASS_KG)
    for component in actor.actor.get_components():
        if isinstance(component, sapien.physx.PhysxRigidDynamicComponent):
            component.max_depenetration_velocity = (
                LOOSE_PROP_MAX_DEPENETRATION_VELOCITY_M_S
            )


class restaurant_pass_counter(Base_Task):

    def setup_demo(self, **kwargs):
        kwargs = kwargs.copy()
        left_embodiment_config = kwargs["left_embodiment_config"].copy()
        right_embodiment_config = kwargs["right_embodiment_config"].copy()
        left_embodiment_config["homestate"] = [LEFT_HOME_STATE.copy(), RIGHT_HOME_STATE.copy()]
        right_embodiment_config["homestate"] = [LEFT_HOME_STATE.copy(), RIGHT_HOME_STATE.copy()]
        kwargs["left_embodiment_config"] = left_embodiment_config
        kwargs["right_embodiment_config"] = right_embodiment_config
        kwargs["table_texture_override"] = "custom/restaurant_dark_grid_10cm"
        super()._init_task_env_(**kwargs)

    def load_actors(self):
        self.microwave = create_sapien_urdf_obj(
            scene=self,
            pose=sapien.Pose([-0.23, 0.20, 0.763], [0.707, 0, 0, 0.707]),
            modelname="044_microwave_2",
            modelid=7167,
            fix_root_link=True,
        )

        self.tray = create_actor(
            scene=self,
            pose=sapien.Pose([0.15, -0.02, 0.741], [0.706527, 0.706483, -0.0291356, -0.0291767]),
            modelname="008_tray",
            convex=True,
            model_id=1,
            is_static=True,
        )

        self.hamburg = create_actor(
            scene=self,
            pose=sapien.Pose([-0.20, -0.25, 0.741], [0.5, 0.5, 0.5, 0.5]),
            modelname="006_hamburg",
            convex=True,
            model_id=0,
        )
        _configure_loose_prop(self.hamburg)

        self.frenchfries = create_actor(
            scene=self,
            pose=sapien.Pose([0.03, -0.27, 0.741], [1.0, 0.0, 0.0, 0.0]),
            modelname="005_french-fries",
            convex=True,
            model_id=0,
        )
        _configure_loose_prop(self.frenchfries)

        self.coke_can = create_actor(
            scene=self,
            pose=sapien.Pose([0.27, -0.24, 0.741], [0.707225, 0.706849, -0.0100455, -0.00982061]),
            modelname="071_can",
            convex=True,
            model_id=3,
        )
        _configure_loose_prop(self.coke_can)

        self.bell = create_actor(
            scene=self,
            pose=sapien.Pose([0.30, 0.20, 0.741], [0.5, 0.5, 0.5, 0.5]),
            modelname="050_bell",
            convex=True,
            model_id=0,
            is_static=True,
        )
