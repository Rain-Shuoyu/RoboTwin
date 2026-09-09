from .restaurant_pass_counter import restaurant_pass_counter
from .utils import create_actor
import sapien


class restaurant_pass_counter_bread_station(restaurant_pass_counter):

    def load_actors(self):
        super().load_actors()

        self.breadbasket = create_actor(
            scene=self,
            pose=sapien.Pose([-0.23, 0.20, 1.134], [0.5, 0.5, 0.5, 0.5]),
            modelname="076_breadbasket",
            convex=True,
            model_id=0,
        )
        self.breadbasket.set_mass(0.12)

        self.bread = [
            create_actor(
                scene=self,
                pose=sapien.Pose([-0.38, -0.10, 0.741], [0.690346, 0.690346, 0.153046, 0.153046]),
                modelname="075_bread",
                convex=True,
                model_id=0,
            ),
            create_actor(
                scene=self,
                pose=sapien.Pose([0.40, -0.08, 0.741], [0.683013, 0.683013, -0.183013, -0.183013]),
                modelname="075_bread",
                convex=True,
                model_id=1,
            ),
        ]
        for bread in self.bread:
            bread.set_mass(0.04)
