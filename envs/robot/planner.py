import mplib.planner
import mplib
import numpy as np
import pdb
import traceback
import numpy as np
import toppra as ta
from mplib.sapien_utils import SapienPlanner, SapienPlanningWorld
import transforms3d as t3d
import envs._GLOBAL_CONFIGS as CONFIGS


def _interpolate_pose(start, end, fraction):
    """Piecewise Cartesian interpolation with a shortest quaternion arc."""
    start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
    a, b = start[3:] / np.linalg.norm(start[3:]), end[3:] / np.linalg.norm(end[3:])
    dot = np.dot(a, b)
    if dot < 0:
        b, dot = -b, -dot
    angle = np.arccos(np.clip(dot, -1., 1.))
    q = ((1 - fraction) * a + fraction * b if angle < 1e-7 else
         (np.sin((1 - fraction) * angle) * a + np.sin(fraction * angle) * b) / np.sin(angle))
    return np.concatenate(((1 - fraction) * start[:3] + fraction * end[:3], q / np.linalg.norm(q)))


def _time_parameterize_joint_path(joints, progress, velocity_limits, acceleration_limits, *, speed_scale=1., dt=1 / 250):
    """Slow one continuous branch until every sampled drive respects limits."""
    joints, progress = np.asarray(joints, dtype=float), np.asarray(progress, dtype=float)
    vmax = np.asarray(velocity_limits, dtype=float) * speed_scale
    amax = np.asarray(acceleration_limits, dtype=float) * speed_scale
    if (joints.ndim != 2 or joints.shape[0] < 2 or not np.all(np.isfinite(joints))
            or progress.shape != (len(joints),) or np.any(vmax <= 0) or np.any(amax <= 0)
            or not np.all(np.isfinite(vmax)) or not np.all(np.isfinite(amax))):
        raise ValueError("invalid path or joint dynamics limits")
    count = max(3, len(joints))
    for _ in range(12):
        t = np.linspace(0., 1., count)
        # Zero start/end derivatives, with the original joint polyline retained.
        parameter = (10 * t ** 3 - 15 * t ** 4 + 6 * t ** 5) * (len(joints) - 1)
        coordinates = np.arange(len(joints))
        positions = np.stack([np.interp(parameter, coordinates, joints[:, i])
                              for i in range(joints.shape[1])], axis=1)
        positions[-1] = joints[-1]
        velocity = np.diff(positions, axis=0, prepend=positions[:1]) / dt
        velocity[-1] = 0.
        acceleration = np.diff(velocity, axis=0, prepend=np.zeros_like(velocity[:1])) / dt
        ratio = max(float(np.max(np.abs(velocity) / vmax)),
                    float(np.sqrt(np.max(np.abs(acceleration) / amax))))
        if ratio <= 1.:
            return positions, velocity, np.interp(parameter, coordinates, progress)
        count = int(np.ceil((count - 1) * ratio * 1.05)) + 1
        if count > 1000000:
            break
    raise ValueError("path cannot be time parameterized within execution bounds")


def _planning_table_world(robot_origin_pose, frame_bias, planning_table):
    """Express a scene table box in the same arm frame used for target poses."""
    if planning_table is None:
        return {"cuboid": {"table": {
            "dims": [0.7, 2, 0.04],
            "pose": [robot_origin_pose.p[1], 0.0, 0.74 - robot_origin_pose.p[2], 1, 0, 0, 0.0],
        }}}
    # The inverse base rotation followed by frame_bias matches plan_path/plan_batch.
    inverse_q = np.asarray(robot_origin_pose.q) * [1, -1, -1, -1]
    delta = np.asarray(planning_table["pose"][:3]) - robot_origin_pose.p
    vector = inverse_q[1:]
    center = delta + 2 * np.cross(vector, np.cross(vector, delta) + inverse_q[0] * delta)
    center += np.asarray(frame_bias)
    table_q = np.asarray(planning_table["pose"][3:])
    quaternion = np.concatenate((
        [inverse_q[0] * table_q[0] - np.dot(vector, table_q[1:])],
        inverse_q[0] * table_q[1:] + table_q[0] * vector + np.cross(vector, table_q[1:]),
    ))
    return {"cuboid": {"table": {
        "dims": list(planning_table["dims"]), "pose": center.tolist() + quaternion.tolist(),
    }}}


try:
    # ********************** CuroboPlanner (optional) **********************
    from curobo.geom.types import WorldConfig
    from curobo.types.math import Pose as CuroboPose
    import time
    from curobo.types.robot import JointState
    from curobo.wrap.reacher.motion_gen import (
        MotionGen,
        MotionGenConfig,
        MotionGenPlanConfig,
        PoseCostMetric,
    )
    from curobo.util import logger
    import torch
    import yaml
    from curobo.util import logger
    logger.setup_logger(level="error", logger_name="curobo")

    class CuroboPlanner:

        def __init__(
            self,
            robot_origion_pose,
            active_joints_name,
            all_joints,
            yml_path=None,
            planning_table=None,
        ):
            super().__init__()
            ta.setup_logging("CRITICAL")  # hide logging
            logger.setup_logger(level="error", logger_name="'curobo")

            if yml_path != None:
                self.yml_path = yml_path
            else:
                raise ValueError("[Planner.py]: CuroboPlanner yml_path is None!")
            self.robot_origion_pose = robot_origion_pose
            self.active_joints_name = active_joints_name
            self.all_joints = all_joints

            # translate from baselink to arm's base
            with open(self.yml_path, "r") as f:
                yml_data = yaml.safe_load(f)
            self.frame_bias = yml_data["planner"]["frame_bias"]

            # Scene-supplied table replaces only the historical default table box.
            world_config = _planning_table_world(robot_origion_pose, self.frame_bias, planning_table)
            motion_gen_config = MotionGenConfig.load_from_robot_config(
                self.yml_path,
                world_config,
                interpolation_dt=1 / 250,
                num_trajopt_seeds=1,
            )

            self.motion_gen = MotionGen(motion_gen_config)
            self.motion_gen.warmup()
            motion_gen_config = MotionGenConfig.load_from_robot_config(
                self.yml_path,
                world_config,
                interpolation_dt=1 / 250,
                num_trajopt_seeds=1,
                num_graph_seeds=1,
            )
            self.motion_gen_batch = MotionGen(motion_gen_config)
            self.motion_gen_batch.warmup(batch=CONFIGS.ROTATE_NUM)

        def update_planning_table(self, robot_origin_pose, planning_table):
            self.robot_origion_pose = robot_origin_pose
            world = WorldConfig.from_dict(_planning_table_world(robot_origin_pose, self.frame_bias, planning_table))
            self.motion_gen.update_world(world)
            self.motion_gen_batch.update_world(world)

        def _joint_state(self, positions):
            return JointState.from_position(
                self.motion_gen.tensor_args.to_device(np.asarray(positions)).reshape(1, -1),
                joint_names=self.active_joints_name,
            )

        def _arm_positions(self, full_positions):
            return np.asarray([full_positions[self.all_joints.index(name)]
                               for name in self.active_joints_name], dtype=float)

        def plan_joint_path(self, curr_joint_pos, target_joint_positions, *, joint_names):
            if set(joint_names) != set(self.active_joints_name):
                return {"status": "Fail", "reason": "joint_names_mismatch"}
            goal = np.asarray([target_joint_positions[list(joint_names).index(name)]
                               for name in self.active_joints_name], dtype=float)
            result = self.motion_gen.plan_single_js(
                self._joint_state(self._arm_positions(curr_joint_pos)), self._joint_state(goal),
                MotionGenPlanConfig(max_attempts=10),
            )
            if not result.success.item():
                return {"status": "Fail", "reason": "joint_planning_failed"}
            return {"status": "Success",
                    "position": np.asarray(result.interpolated_plan.position.cpu()),
                    "velocity": np.asarray(result.interpolated_plan.velocity.cpu())}

        def _path_pose_in_base(self, pose, arms_tag):
            p, q = self._trans_from_world_to_base(
                np.concatenate((self.robot_origion_pose.p, self.robot_origion_pose.q)),
                np.concatenate((pose.p, pose.q)),
            )
            rotation = np.eye(3)
            if "aloha-agilex" in self.yml_path:
                rotation = t3d.axangles.axangle2mat([0, 0, 1], -.02 if arms_tag == "left" else -.01)
            return np.concatenate((rotation @ (p + self.frame_bias),
                t3d.quaternions.mat2quat(rotation @ t3d.quaternions.quat2mat(q))))

        def forward_kinematics(self, joint_positions, *, joint_names, arms_tag):
            positions = np.asarray(joint_positions, dtype=float)
            if positions.ndim == 1:
                positions = positions[None, :]
            ordered = positions[:, [list(joint_names).index(name) for name in self.active_joints_name]]
            state = JointState.from_position(self.motion_gen.tensor_args.to_device(ordered),
                                             joint_names=self.active_joints_name)
            measured = self.motion_gen.compute_kinematics(state)
            base_positions = np.asarray(measured.ee_pos_seq.cpu()).reshape(-1, 3)
            base_quaternions = np.asarray(measured.ee_quat_seq.cpu()).reshape(-1, 4)
            origin_rotation = t3d.quaternions.quat2mat(self.robot_origion_pose.q)
            correction = np.eye(3)
            if "aloha-agilex" in self.yml_path:
                correction = t3d.axangles.axangle2mat([0, 0, 1], -.02 if arms_tag == "left" else -.01)
            output = []
            for p, q in zip(base_positions, base_quaternions):
                world_p = origin_rotation @ (correction.T @ p - self.frame_bias) + self.robot_origion_pose.p
                world_q = t3d.quaternions.mat2quat(origin_rotation @ correction.T @ t3d.quaternions.quat2mat(q))
                output.append(np.concatenate((world_p, world_q)))
            return np.asarray(output)

        def plan_tcp_path(self, curr_joint_pos, target_gripper_pose_list, *, arms_tag,
                          speed_scale=1., position_tolerance_m=.005,
                          orientation_tolerance_rad=.02, max_joint_step_rad=.2, pauses=(),
                          tcp_bias=0., tcp_delta=None):
            """Follow a Cartesian polyline; never freely plan its endpoints."""
            current = self._arm_positions(curr_joint_pos)
            limits = self.motion_gen.kinematics.get_joint_limits()
            lower, upper = np.asarray(limits.position.cpu())
            velocity_limit = np.min(np.abs(np.asarray(limits.velocity.cpu())), axis=0)
            acceleration_limit = np.min(np.abs(np.asarray(limits.acceleration.cpu())), axis=0)
            raw_state = self.motion_gen.compute_kinematics(self._joint_state(current))
            delta = np.eye(3) if tcp_delta is None else np.asarray(tcp_delta)

            def tcp_pose(position, quaternion):
                rotation = t3d.quaternions.quat2mat(quaternion) @ delta
                return np.concatenate((position + rotation @ np.array([tcp_bias, 0., 0.]),
                                       t3d.quaternions.mat2quat(rotation)))

            start = tcp_pose(np.asarray(raw_state.ee_pos_seq.cpu()).reshape(3),
                             np.asarray(raw_state.ee_quat_seq.cpu()).reshape(4))
            references = [start] + [self._path_pose_in_base(pose, arms_tag)
                                    for pose in target_gripper_pose_list]
            solutions, progress = [current], [0.]
            completed_points = 0
            for point_index, (a, b) in enumerate(zip(references, references[1:])):
                angle = 2 * np.arccos(np.clip(abs(np.dot(a[3:], b[3:])), 0, 1))
                divisions = max(1, int(np.ceil(np.linalg.norm(b[:3] - a[:3]) / (position_tolerance_m / 2))),
                                int(np.ceil(angle / (orientation_tolerance_rad / 2))))
                for division in range(1, divisions + 1):
                    fraction = division / divisions
                    target = _interpolate_pose(a, b, fraction)
                    rotation = t3d.quaternions.quat2mat(target[3:])
                    endlink_target = np.concatenate((target[:3] - rotation @ np.array([tcp_bias, 0., 0.]),
                        t3d.quaternions.mat2quat(rotation @ delta.T)))
                    seed = self.motion_gen.tensor_args.to_device(current).reshape(1, -1)
                    solved = self.motion_gen.ik_solver.solve_single(
                        CuroboPose.from_list(endlink_target.tolist()), retract_config=seed,
                        seed_config=seed.reshape(1, 1, -1), return_seeds=1,
                    )
                    if not bool(solved.success.item()):
                        return {"status": "Fail", "reason": "ik_failed", "planned_path_points": completed_points}
                    solution = np.asarray(solved.solution.cpu()).reshape(-1, len(current))[0]
                    if (not np.all(np.isfinite(solution)) or np.any(solution < lower) or np.any(solution > upper)
                            or np.max(np.abs(solution - current)) > max_joint_step_rad):
                        return {"status": "Fail", "reason": "joint_limit_or_jump", "planned_path_points": completed_points}
                    current = solution
                    solutions.append(current)
                    progress.append(point_index + fraction)
                completed_points += 1
            try:
                holds = {}
                for pause in pauses:
                    marker = int(pause["after_pose_index"]) + 1
                    duration = float(pause["duration_s"])
                    if marker < 1 or marker >= len(references) or not np.isfinite(duration) or duration < 0:
                        raise ValueError("invalid pause")
                    holds[marker] = holds.get(marker, 0.) + duration
                # Splitting at pauses makes their exact path points rest
                # states. The held samples are part of the same dense action.
                cuts = sorted(set(holds) | {len(references) - 1})
                parts, previous, pause_intervals, sample_count = [], 0, [], 0
                for marker in cuts:
                    stop = next(i for i, value in enumerate(progress) if abs(value - marker) < 1e-9)
                    p, v, s = _time_parameterize_joint_path(
                        solutions[previous:stop + 1], progress[previous:stop + 1],
                        velocity_limit, acceleration_limit, speed_scale=speed_scale)
                    parts.append((p, v, s))
                    sample_count += len(p)
                    count = int(np.ceil(holds.get(marker, 0.) * 250))
                    if count > 1000000:
                        raise ValueError("pause exceeds execution bounds")
                    if count:
                        pause_intervals.append({"after_pose_index": marker - 1,
                            "start_step": sample_count, "end_step": sample_count + count,
                            "duration_s": holds[marker]})
                        parts.append((np.repeat(p[-1:], count, axis=0),
                                      np.zeros((count, len(current))), np.full(count, marker)))
                        sample_count += count
                    previous = stop
                positions = np.concatenate([part[0] for part in parts])
                velocities = np.concatenate([part[1] for part in parts])
                dense_progress = np.concatenate([part[2] for part in parts])
            except ValueError:
                return {"status": "Fail", "reason": "time_parameterization_failed", "planned_path_points": completed_points}
            # Check every actual controller sample, including joint interpolation
            # between IK solutions, against its point on the Cartesian polyline.
            state = JointState.from_position(self.motion_gen.tensor_args.to_device(positions),
                                             joint_names=self.active_joints_name)
            actual = self.motion_gen.compute_kinematics(state)
            actual_p = np.asarray(actual.ee_pos_seq.cpu()).reshape(-1, 3)
            actual_q = np.asarray(actual.ee_quat_seq.cpu()).reshape(-1, 4)
            for i, value in enumerate(dense_progress):
                index = min(int(value), len(references) - 2)
                expected = _interpolate_pose(references[index], references[index + 1], value - index)
                measured_tcp = tcp_pose(actual_p[i], actual_q[i])
                orientation_error = 2 * np.arccos(np.clip(abs(np.dot(expected[3:], measured_tcp[3:])), 0, 1))
                if (np.linalg.norm(expected[:3] - measured_tcp[:3]) > position_tolerance_m
                        or orientation_error > orientation_tolerance_rad):
                    return {"status": "Fail", "reason": "path_fk_residual_exceeded", "planned_path_points": completed_points}
            return {"status": "Success", "position": positions, "velocity": velocities,
                    "path_progress": dense_progress, "planned_path_points": completed_points,
                    "pause_intervals": pause_intervals}

        def plan_path(
            self,
            curr_joint_pos,
            target_gripper_pose,
            constraint_pose=None,
            arms_tag=None,
        ):  
            world_base_pose = np.concatenate([
                np.array(self.robot_origion_pose.p),
                np.array(self.robot_origion_pose.q),
            ])
            world_target_pose = np.concatenate([np.array(target_gripper_pose.p), np.array(target_gripper_pose.q)])
            target_pose_p, target_pose_q = self._trans_from_world_to_base(world_base_pose, world_target_pose)
            if not ("aloha-agilex" in self.yml_path):
                target_pose_p[0] += self.frame_bias[0]
                target_pose_p[1] += self.frame_bias[1]
                target_pose_p[2] += self.frame_bias[2]
            else: # patch for aloha-agilex
                T_target = t3d.affines.compose(target_pose_p, t3d.quaternions.quat2mat(target_pose_q), [1, 1, 1])
                T_bias = t3d.affines.compose(self.frame_bias, np.eye(3), [1, 1, 1])

                if arms_tag == "left":
                    rot = t3d.axangles.axangle2mat([0, 0, 1], -0.02)
                elif arms_tag == "right":
                    rot = t3d.axangles.axangle2mat([0, 0, 1], -0.01)
                else:
                    raise ValueError(f"Invalid arms_tag: {arms_tag}")

                T_rot = t3d.affines.compose([0, 0, 0], rot, [1, 1, 1])
                T_new = T_rot @ T_bias @ T_target
                target_pose_p = T_new[:3, 3]
                target_pose_q = t3d.quaternions.mat2quat(T_new[:3, :3])

            goal_pose_of_ee = CuroboPose.from_list(list(target_pose_p) + list(target_pose_q))
            joint_indices = [self.all_joints.index(name) for name in self.active_joints_name if name in self.all_joints]
            joint_angles = [curr_joint_pos[index] for index in joint_indices]
            joint_angles = [round(angle, 5) for angle in joint_angles]  # avoid the precision problem
            start_joint_states = JointState.from_position(
                torch.tensor(joint_angles).cuda().reshape(1, -1),
                joint_names=self.active_joints_name,
            )
            # plan
            plan_config = MotionGenPlanConfig(max_attempts=10)
            if constraint_pose is not None:
                pose_cost_metric = PoseCostMetric(
                    hold_partial_pose=True,
                    hold_vec_weight=self.motion_gen.tensor_args.to_device(constraint_pose),
                )
                plan_config.pose_cost_metric = pose_cost_metric

            result = self.motion_gen.plan_single(start_joint_states, goal_pose_of_ee, plan_config)

            # output
            res_result = dict()
            if result.success.item() == False:
                res_result["status"] = "Fail"
                return res_result
            else:
                res_result["status"] = "Success"
                res_result["position"] = np.array(result.interpolated_plan.position.to("cpu"))
                res_result["velocity"] = np.array(result.interpolated_plan.velocity.to("cpu"))
                return res_result

        def plan_batch(
            self,
            curr_joint_pos,
            target_gripper_pose_list,
            constraint_pose=None,
            arms_tag=None,
        ):
            """
            Plan a batch of trajectories for multiple target poses.

            Input:
                - curr_joint_pos: List of current joint angles (1 x n)
                - target_gripper_pose_list: List of target poses [sapien.Pose, sapien.Pose, ...]

            Output:
                - result['status']: numpy array of string values indicating "Success"/"Fail" for each pose
                - result['position']: numpy array of joint positions with shape (n x m x l)
                  where n is number of target poses, m is number of waypoints, l is number of joints
                - result['velocity']: numpy array of joint velocities with same shape as position
            """

            num_poses = len(target_gripper_pose_list)
            # transformation from world to arm's base
            world_base_pose = np.concatenate([
                np.array(self.robot_origion_pose.p),
                np.array(self.robot_origion_pose.q),
            ])
            poses_list = []
            for target_gripper_pose in target_gripper_pose_list:
                world_target_pose = np.concatenate([np.array(target_gripper_pose.p), np.array(target_gripper_pose.q)])
                base_target_pose_p, base_target_pose_q = self._trans_from_world_to_base(world_base_pose, world_target_pose)

                if not ("aloha-agilex" in self.yml_path):
                    base_target_pose_p[0] += self.frame_bias[0]
                    base_target_pose_p[1] += self.frame_bias[1]
                    base_target_pose_p[2] += self.frame_bias[2]
                else: # patch for aloha-agilex
                    T_target = t3d.affines.compose(base_target_pose_p, t3d.quaternions.quat2mat(base_target_pose_q), [1, 1, 1])
                    T_bias = t3d.affines.compose(self.frame_bias, np.eye(3), [1, 1, 1])

                    if arms_tag == "left":
                        rot = t3d.axangles.axangle2mat([0, 0, 1], -0.02)
                    elif arms_tag == "right":
                        rot = t3d.axangles.axangle2mat([0, 0, 1], -0.01)
                    else:
                        raise ValueError(f"Invalid arms_tag: {arms_tag}")

                    T_rot = t3d.affines.compose([0, 0, 0], rot, [1, 1, 1])
                    T_new = T_rot @ T_bias @ T_target
                    base_target_pose_p = T_new[:3, 3]
                    base_target_pose_q = t3d.quaternions.mat2quat(T_new[:3, :3])

                base_target_pose_list = list(base_target_pose_p) + list(base_target_pose_q)
                poses_list.append(base_target_pose_list)

            poses_cuda = torch.tensor(poses_list, dtype=torch.float32).cuda()
            goal_pose_of_ee = CuroboPose(poses_cuda[:, :3], poses_cuda[:, 3:])
            joint_indices = [self.all_joints.index(name) for name in self.active_joints_name if name in self.all_joints]
            joint_angles = [curr_joint_pos[index] for index in joint_indices]
            joint_angles = [round(angle, 5) for angle in joint_angles]  # avoid the precision problem
            joint_angles_cuda = (torch.tensor(joint_angles, dtype=torch.float32).cuda().reshape(1, -1))
            joint_angles_cuda = torch.cat([joint_angles_cuda] * num_poses, dim=0)
            start_joint_states = JointState.from_position(joint_angles_cuda, joint_names=self.active_joints_name)
            # plan
            plan_config = MotionGenPlanConfig(max_attempts=10)
            if constraint_pose is not None:
                pose_cost_metric = PoseCostMetric(
                    hold_partial_pose=True,
                    hold_vec_weight=self.motion_gen.tensor_args.to_device(constraint_pose),
                )
                plan_config.pose_cost_metric = pose_cost_metric

            try:
                result = self.motion_gen_batch.plan_batch(start_joint_states, goal_pose_of_ee, plan_config)
            except Exception as e:
                return {"status": ["Failure" for i in range(10)]}

            # output
            res_result = dict()
            # Convert boolean success values to "Success"/"Failure" strings
            success_array = result.success.cpu().numpy()
            status_array = np.array(["Success" if s else "Failure" for s in success_array], dtype=object)
            res_result["status"] = status_array

            if np.all(res_result["status"] == "Failure"):
                return res_result

            res_result["position"] = np.array(result.interpolated_plan.position.to("cpu"))
            res_result["velocity"] = np.array(result.interpolated_plan.velocity.to("cpu"))
            return res_result

        def plan_grippers(self, now_val, target_val):
            num_step = 200
            dis_val = target_val - now_val
            step = dis_val / num_step
            res = {}
            vals = np.linspace(now_val, target_val, num_step)
            res["num_step"] = num_step
            res["per_step"] = step
            res["result"] = vals
            return res

        def _trans_from_world_to_base(self, base_pose, target_pose):
            '''
                transform target pose from world frame to base frame
                base_pose: np.array([x, y, z, qw, qx, qy, qz])
                target_pose: np.array([x, y, z, qw, qx, qy, qz])
            '''
            base_p, base_q = base_pose[0:3], base_pose[3:]
            target_p, target_q = target_pose[0:3], target_pose[3:]
            rel_p = target_p - base_p
            wRb = t3d.quaternions.quat2mat(base_q)
            wRt = t3d.quaternions.quat2mat(target_q)
            result_p = wRb.T @ rel_p
            result_q = t3d.quaternions.mat2quat(wRb.T @ wRt)
            return result_p, result_q
    
except Exception as e:
    print('[planner.py]: Something wrong happened when importing CuroboPlanner! Please check if Curobo is installed correctly. If the problem still exists, you can install Curobo from https://github.com/NVlabs/curobo manually.')
    print('Exception traceback:')
    traceback.print_exc()


# ********************** MplibPlanner **********************
class MplibPlanner:
    # links=None, joints=None
    def __init__(
        self,
        urdf_path,
        srdf_path,
        move_group,
        robot_origion_pose,
        robot_entity,
        planner_type="mplib_RRT",
        scene=None,
    ):
        super().__init__()
        ta.setup_logging("CRITICAL")  # hide logging

        links = [link.get_name() for link in robot_entity.get_links()]
        joints = [joint.get_name() for joint in robot_entity.get_active_joints()]

        if scene is None:
            self.planner = mplib.Planner(
                urdf=urdf_path,
                srdf=srdf_path,
                move_group=move_group,
                user_link_names=links,
                user_joint_names=joints,
                use_convex=False,
            )
            self.planner.set_base_pose(robot_origion_pose)
        else:
            planning_world = SapienPlanningWorld(scene, [robot_entity])
            self.planner = SapienPlanner(planning_world, move_group)

        self.planner_type = planner_type
        self.plan_step_lim = 2500
        self.TOPP = self.planner.TOPP

    def show_info(self):
        print("joint_limits", self.planner.joint_limits)
        print("joint_acc_limits", self.planner.joint_acc_limits)

    def plan_pose(
        self,
        now_qpos,
        target_pose,
        use_point_cloud=False,
        use_attach=False,
        arms_tag=None,
        try_times=2,
        log=True,
    ):
        result = {}
        result["status"] = "Fail"

        now_try_times = 1
        while result["status"] != "Success" and now_try_times < try_times:
            result = self.planner.plan_pose(
                goal_pose=target_pose,
                current_qpos=np.array(now_qpos),
                time_step=1 / 250,
                planning_time=5,
                # rrt_range=0.05
                # =================== mplib 0.1.1 ===================
                # use_point_cloud=use_point_cloud,
                # use_attach=use_attach,
                # planner_name="RRTConnect"
            )
            now_try_times += 1

        if result["status"] != "Success":
            if log:
                print(f"\n {arms_tag} arm planning failed ({result['status']}) !")
        else:
            n_step = result["position"].shape[0]
            if n_step > self.plan_step_lim:
                if log:
                    print(f"\n {arms_tag} arm planning wrong! (step = {n_step})")
                result["status"] = "Fail"

        return result

    def plan_screw(
        self,
        now_qpos,
        target_pose,
        use_point_cloud=False,
        use_attach=False,
        arms_tag=None,
        log=False,
    ):
        """
        Interpolative planning with screw motion.
        Will not avoid collision and will fail if the path contains collision.
        """
        result = self.planner.plan_screw(
            goal_pose=target_pose,
            current_qpos=now_qpos,
            time_step=1 / 250,
            # =================== mplib 0.1.1 ===================
            # use_point_cloud=use_point_cloud,
            # use_attach=use_attach,
        )

        # plan fail
        if result["status"] != "Success":
            if log:
                print(f"\n {arms_tag} arm planning failed ({result['status']}) !")
            # return result
        else:
            n_step = result["position"].shape[0]
            # plan step lim
            if n_step > self.plan_step_lim:
                if log:
                    print(f"\n {arms_tag} arm planning wrong! (step = {n_step})")
                result["status"] = "Fail"

        return result

    def plan_path(
        self,
        now_qpos,
        target_pose,
        use_point_cloud=False,
        use_attach=False,
        arms_tag=None,
        log=True,
    ):
        """
        Interpolative planning with screw motion.
        Will not avoid collision and will fail if the path contains collision.
        """
        if self.planner_type == "mplib_RRT":
            result = self.plan_pose(
                now_qpos,
                target_pose,
                use_point_cloud,
                use_attach,
                arms_tag,
                try_times=10,
                log=log,
            )
        elif self.planner_type == "mplib_screw":
            result = self.plan_screw(now_qpos, target_pose, use_point_cloud, use_attach, arms_tag, log)

        return result

    def plan_grippers(self, now_val, target_val):
        num_step = 200  # TODO
        dis_val = target_val - now_val
        per_step = dis_val / num_step
        res = {}
        vals = np.linspace(now_val, target_val, num_step)
        res["num_step"] = num_step
        res["per_step"] = per_step  # dis per step
        res["result"] = vals
        return res
