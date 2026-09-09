from ._base_task import Base_Task
from .utils import *
import sapien


class restaurant_pass_counter(Base_Task):

    def setup_demo(self, **kwargs):
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
        self.hamburg.set_mass(0.05)

        self.frenchfries = create_actor(
            scene=self,
            pose=sapien.Pose([0.03, -0.27, 0.741], [1.0, 0.0, 0.0, 0.0]),
            modelname="005_french-fries",
            convex=True,
            model_id=0,
        )
        self.frenchfries.set_mass(0.05)

        self.coke_can = create_actor(
            scene=self,
            pose=sapien.Pose([0.27, -0.24, 0.741], [0.707225, 0.706849, -0.0100455, -0.00982061]),
            modelname="071_can",
            convex=True,
            model_id=3,
        )
        self.coke_can.set_mass(0.05)

        self.bell = create_actor(
            scene=self,
            pose=sapien.Pose([0.42, 0.20, 0.741], [0.5, 0.5, 0.5, 0.5]),
            modelname="050_bell",
            convex=True,
            model_id=0,
            is_static=True,
        )
