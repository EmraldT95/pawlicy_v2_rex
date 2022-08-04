import numpy as np

class WalkAlongX(object):
    """Task to walk along a straight line (x-axis)"""
    def __init__(self,
                #forward_reward_cap: float = float("inf"),
                velocity_weight: float = 32,
                distance_weight: float = 0.02,
                forward_weight : float = 1,
                step_weight : float = 0.223,
                # energy_weight=0.0005,
                displacement_weight : float = 800,
                shake_weight: float = 1,
                drift_weight: float = 50,
                orientation_weight : float = 1,
                pose_weight : float = 1,
                action_cost_weight: float = 0.0004,
                # deviation_weight: float = 1,
                enable_roll_limit : bool = True,
                healthy_roll_limit : float = np.pi * 1/2,
                # roll_threshold: float = np.pi * 1/2,
                # pitch_threshold: float = 0.8,
                enable_z_limit: bool = True,
                healthy_z_limit: float = 0.15,
                # healthy_reward=1.0,
                ):
        """Initializes the task."""

        #self._forward_reward_cap = forward_reward_cap
        self._action_cost_weight = action_cost_weight
        self._velocity_weight = velocity_weight
        self._distance_weight = distance_weight
        self._forward_weight = forward_weight
        self._shake_weight = shake_weight
        self._drift_weight = drift_weight
        self._step_weight = step_weight
        self._orientation_weight = orientation_weight
        self._pose_weight = pose_weight
        self._displacement_weight = displacement_weight
        # self._deviation_weight = deviation_weight
        #self.roll_threshold = roll_threshold
        self.enable_z_limit = enable_z_limit
        self.healthy_z_limit = healthy_z_limit
        # self.healthy_reward = healthy_reward
        #self.pitch_threshold = pitch_threshold
        self.enable_roll_limit = enable_roll_limit
        self.healthy_roll_limit = healthy_roll_limit

        self.step_counter = 0
        self.max_vel = 0
        self._target_pos =  np.array([100, 0, 1])

        self.max_ori = float("-inf")
        self.min_ori = float("inf")

    def __call__(self, env):
        return self.reward(env)

    def reset(self, env):
        """Resets the internal state of the task."""
        self._env = env
        self._init_base_pos = np.array(env.robot.GetBasePosition())
        
        self._current_base_pos = self._init_base_pos
        self._last_base_pos = self._current_base_pos
        
        self._current_base_vel = env.robot.GetBaseVelocity()
        # self._alive_time_reward = 0
        # self._cumulative_displacement = 0
        self._last_action = env.robot.last_action
        
        #self._current_base_ori_euler = self.GetRawOrientation(env)
        self._current_base_ori_euler = env.robot.GetTrueBaseRollPitchYaw()
        self._init_base_ori_euler = self._current_base_ori_euler
        #print(f"Initial Orienttaion {self._init_base_ori_euler}")
        self.step_counter = 0

        
    def update(self, env):
        """Updates the internal state of the task.
        Evoked after call to a1.A1.Step(), ie after action takes effect in simulation
        """
        self._last_base_pos = self._current_base_pos
        self._current_base_pos = np.array(env.robot.GetBasePosition())

        self._current_base_vel = env.robot.GetBaseVelocity()
        # self._alive_time_reward = env.get_time_since_reset()
        self._last_action = env.last_action
        
        #self._current_base_ori_euler = self.GetRawOrientation(env)
        self._current_base_ori_euler = env.robot.GetTrueBaseRollPitchYaw()

        self.step_counter += self._step_weight

        if self._current_base_vel[0] > self.max_vel:
            self.max_vel = self._current_base_vel[0]
            #print(f"Maximum Velocity of base {self.max_vel}")

    def done(self, env):
        """Checks if the episode is over.

            If the robot base becomes unstable (based on orientation), the episode
            terminates early.
        """
        rot_quat = env.robot.GetBaseOrientation()
        rot_mat = env.pybullet_client.getMatrixFromQuaternion(rot_quat)
        return rot_mat[-1] < 0.85


    def reward(self, env):

        # bug : -ve velocity along y is rewarded
        #velocity_reward = np.dot([1, -1, 0], self._current_base_vel)
        # velocity_reward = self._velocity_weight * self._current_base_vel[0]# if self._current_base_vel[0] < 0.4 else 0.4
        # forward_reward = self._forward_weight * (self._current_base_pos[0] - self._init_base_pos[0])
        displacement_reward = self._displacement_weight * (self._current_base_pos[0] - self._last_base_pos[0])

        # y_velocity_reward = -abs(self._current_base_vel[1])
        action_reward = - self._action_cost_weight * np.linalg.norm(self._last_action)
        
        
        #velocity_reward = self._velocity_weight * self._current_base_vel[0]
        #distance_reward = - self._distance_weight * np.linalg.norm(self._target_pos - self._current_base_pos)
        #displacement_reward = self._current_base_pos - self._last_base_pos)
        #forward_reward =  self._forward_weight * (self._current_base_pos[0] - self._init_base_pos[0])
        #pose_reward = self._pose_weight * (self._current_base_pos[2] if self._current_base_pos[2] < 0.3 else 0.3)
        #orientation_reward = - self._orientation_weight * np.linalg.norm(env.robot.GetTrueBaseRollPitchYaw() - self._init_base_ori_euler)
        # drift_reward =  - self._drift_weight * (self._current_base_pos[1]) ** 2
        
        # orientation = env.robot.GetBaseOrientation()
        # rot_matrix = env.robot._pybullet_client.getMatrixFromQuaternion(orientation)
        # local_up_vec = rot_matrix[6:]
        # shake_reward = - self._shake_weight * abs(np.dot(np.asarray([1, 1, 0]), np.asarray(local_up_vec)))
    
        rot_quat = env.robot.GetBaseOrientation()
        rot_mat = env.pybullet_client.getMatrixFromQuaternion(rot_quat)
        orientation_reward = self._orientation_weight * rot_mat[-1] ** 2
        # print(f"displacement {displacement_reward} action {action_reward} orientation {orientation_reward} drift {drift_reward}")
        reward = displacement_reward + action_reward + orientation_reward # + self._step_weight 
            #+ shake_reward # + y_velocity_reward + forward_reward + displacement_reward + action_reward \
                  #
        #print("Reward", reward)
        return reward


    @property
    def is_healthy(self):
        # Check for counterclockwise rotation along x-axis and y-axis (in radians)
        if self.enable_roll_limit:
            aug_vec = np.array([1, -1, 1])
            orientation_aug = self._current_base_ori_euler * aug_vec
            if np.any(orientation_aug < -self.healthy_roll_limit) or np.any(orientation_aug > self.healthy_roll_limit):
                return False

        # Isuue - needs to account for heightfield data
        if self.enable_z_limit and (self._current_base_pos[2] < self.healthy_z_limit):
            return False
        # if self.enable_roll_limit and (self._current_base_ori_euler[0] < -self.healthy_roll_limit or \
        #     self._current_base_ori_euler[0] > self.healthy_roll_limit):
        #     return False

        # Checking if robot is in contact with the ground
        # foot_links = env.robot.GetFootLinkIDs()
        # ground = env.get_ground()
        # # Skip the first env step
        # if env.env_step_counter > 0:
        #     robot_ground_contacts = env.pybullet_client.getContactPoints(bodyA=env.robot.quadruped, bodyB=ground["id"])
        #     for contact in robot_ground_contacts:
        #         # Only the toes of the robot should in contact with the ground
        #         if contact[3] not in foot_links:
        #             # print("contact_fail")
        #             return False

            # # The robot shouldn't be flipped, so limit the Roll and Pitch
            # if self._current_base_ori_euler[0] > self.roll_threshold or self._current_base_ori_euler[0] < -self.roll_threshold:
            #     # print("roll_fail")
            #     return False
            # if self._current_base_ori_euler[1] > self.pitch_threshold or self._current_base_ori_euler[1] < -self.pitch_threshold:
            #     # print("pitch_fail")
            #     return False

            # Issue - needs to account for heightfield data
            # if self.enable_z_limit:
            #     # z_upper_limit = self._init_base_pos[2] + self.healthy_z_limit * 2
            #     # z_lower_limit = self._init_base_pos[2] - self.healthy_z_limit
            #     # # Random terrains don't have fixed heights in every run. Hence initial position of
            #     # # the robot is generally higher than usual. Taking that under consideration
            #     # if env.world_dict["ground"]["type"] != "plain":
            #     #     z_lower_limit -= 0.1
            #     # Robot shouldn't be below the limit
            #     if self._current_base_pos[2] < 0.02:
            #         # print("z_fail: Current=", self._current_base_pos[2], ",\tLimit=", z_lower_limit)
            #         return False
        return True