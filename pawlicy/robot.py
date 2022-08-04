import re
import collections
import math
import copy
import numpy as np

from pawlicy.envs import TerrainConstants
from pawlicy.action_filter import ActionFilterButter

# Some constants specific to the URDF file
LINK_NAME_ID_DICT = {
    "trunk" : -1,  "imu_link" : 0,
    "FR_hip_joint" : 1, "FR_upper_shoulder" : 2, "FR_upper_joint" : 3, "FR_lower_joint" : 4, "FR_toe" : 5,
    "FL_hip_joint" : 6, "FL_upper_shoulder" : 7, "FL_upper_joint" : 8, "FL_lower_joint" : 9, "FL_toe" : 10,
    "RR_hip_joint" : 11, "RR_upper_shoulder" : 12, "RR_upper_joint" : 13, "RR_lower_joint" : 14, "RR_toe" : 15,
    "RL_hip_joint" : 16, "RL_upper_shoulder" : 17, "RL_upper_joint" : 18, "RL_lower_joint" : 19, "RL_toe" : 20,
}
JOINT_NAMES = [
    "FR_hip_joint", "FR_upper_joint", "FR_lower_joint",
    "FL_hip_joint", "FL_upper_joint", "FL_lower_joint",
    "RR_hip_joint", "RR_upper_joint", "RR_lower_joint",
    "RL_hip_joint", "RL_upper_joint", "RL_lower_joint",
]
NUM_MOTORS = 12
NUM_LEGS = 4
NUM_MOTORS_PER_LEG = 3
INIT_HIP_ANGLE = 0
INIT_UPPER_ANGLE = 0.9
INIT_LOWER_ANGLE = -1.6
INIT_MOTOR_ANGLES = np.array([INIT_HIP_ANGLE, INIT_UPPER_ANGLE, INIT_LOWER_ANGLE] * NUM_LEGS)
HIP_NAME_PATTERN = re.compile(r"\w+_hip_\w+")
UPPER_NAME_PATTERN = re.compile(r"\w+_upper_\w+")
LOWER_NAME_PATTERN = re.compile(r"\w+_lower_\w+")
TOE_NAME_PATTERN = re.compile(r"\w+_toe\d*")
IMU_NAME_PATTERN = re.compile(r"imu\d*")
MOTOR_NAME_PATTERN = re.compile(r"^(?!imu).*_joint")
SENSOR_NOISE_STDDEV = (0.0, 0.0, 0.0, 0.0, 0.0)
MAX_MOTOR_ANGLE_CHANGE_PER_STEP = 0.2

def MapToMinusPiToPi(angles):
    """Maps a list of angles to [-pi, pi].

      Args:
        angles: A list of angles in rad.
      Returns:
        A list of angle mapped to [-pi, pi].
    """
    mapped_angles = copy.deepcopy(angles)
    for i in range(len(angles)):
        mapped_angles[i] = math.fmod(angles[i], 2 * math.pi)
        if mapped_angles[i] >= math.pi:
            mapped_angles[i] -= 2 * math.pi
        elif mapped_angles[i] < -math.pi:
            mapped_angles[i] += 2 * math.pi
    return mapped_angles

class A1:
    """The A1 robot class. This URDF of the same is available from pybullet"""

    def __init__(self,
                pybullet_client,
                action_repeat=1,
                time_step = 0.01,
                control_latency=0.0,
                motor_kp=1.0,
                motor_kd=0.02,
                observation_noise_stdev=SENSOR_NOISE_STDDEV,
                terrain="plane"):
        """Constructs an A1 robot and reset it to the initial states.
        
        Args:
            pybullet_client: The pybullet client.
            action_repeat: The number of ApplyAction() for each control step.
            time_step: The time step of the simulation.
            control_latency: The latency of the observations (in second) used to
                calculate action. On the real hardware, it is the latency from the motor
                controller, the microcontroller to the host (for eg., Nvidia TX2).
            motor_kp: proportional gain for the accurate motor model.
            motor_kd: derivative gain for the accurate motor model.
            observation_noise_stdev: The standard deviation of a Gaussian noise model
                for the sensor. It should be an array for separate sensors in the
                following order [motor_angle, motor_velocity, motor_torque,
                base_roll_pitch_yaw, base_angular_velocity]
            terrain: The terrain on which the robot is standing.
        """

        self._pybullet_client = pybullet_client
        self._action_repeat = action_repeat
        self._init_base_position = TerrainConstants.ROBOT_INIT_POSITION[terrain]
        self._init_base_orientation = self._pybullet_client.getQuaternionFromEuler([0, 0, 0])
        self._control_latency = control_latency
        self._observation_noise_stdev = observation_noise_stdev
        self._terrain = terrain

        self.num_motors = NUM_MOTORS
        self._motor_direction = [1 for _ in range(self.num_motors)]
        self._trunk = -1
        self._hip_link_ids = []
        self._upper_link_ids = []
        self._lower_link_ids = []
        self._foot_link_ids = []
        self._imu_link_ids = []
        self._motor_link_ids = []
        self._joint_lower_limits = []
        self._joint_upper_limits = []
        self._joint_max_force = []
        self._joint_max_velocity = []
        self._joint_name_to_id = {}
        self._observation_history = collections.deque(maxlen=100)
        self._kp = motor_kp # Proportional gain of motors
        self._kd = motor_kd # Derivative gain of motors
        self.time_step = time_step
        self._step_counter = 0

        self._action_filter = self._BuildActionFilter()
        # reset_time=-1.0 means skipping the reset motion.
        # See Reset for more details.
        self.Reset(reset_time=-1)

    def Terminate(self):
        pass

    def Reset(self, hard_reload=True, default_motor_angles=None, reset_time=3.0):
        """Resets the robot"""

        if hard_reload:
            # Build the robot
            self._robot_id = self._pybullet_client.loadURDF("a1/a1.urdf",
                                                        self._init_base_position,
                                                        self._init_base_orientation,
                                                        useFixedBase=False,
                                                        flags=self._pybullet_client.URDF_USE_SELF_COLLISION)
            self._num_joints = self._pybullet_client.getNumJoints(self._robot_id)
            # self._RemoveDefaultJointDamping() # Remove joint damping
            self._GetJointsInfo() # Get some information about the joints of the robot
        else:
            # Reset the position of the Robot
            self._pybullet_client.resetBasePositionAndOrientation(self._robot_id, self._init_base_position, self._init_base_orientation)
            self._pybullet_client.resetBaseVelocity(self._robot_id, [0, 0, 0], [0, 0, 0])

        # Reset the pose of the Robot
        self.ResetPose()
        self._current_velocity = np.zeros_like(self.GetBaseVelocity())
        self._last_velocity = np.zeros_like(self._current_velocity)
        self._step_counter = 0
        self._last_action = None
        self._observation_history.clear()
        # Perform reset motion within reset_duration if in position control mode.
        # Nothing is performed if in torque control mode for now.
        if reset_time > 0.0:
            pose = INIT_MOTOR_ANGLES
            self.ReceiveObservation()
            for _ in range(100):
                self.ApplyAction(pose)
                self._pybullet_client.stepSimulation()
                self.ReceiveObservation()
                    
        self._ResetActionFilter()
        self.ReceiveObservation()

    def ResetPose(self):
        """
        Resets the pose of the robot to its initial pose
        """

        # Set the angle(in radians) for each joint
        for name, i in zip(JOINT_NAMES, range(len(JOINT_NAMES))):
            self._pybullet_client.resetJointState(self._robot_id,
                                                  self._joint_name_to_id[name],
                                                  INIT_MOTOR_ANGLES[i],
                                                  targetVelocity=0)

    def _BuildActionFilter(self):
        sampling_rate = 1 / (self.time_step * self._action_repeat)
        a_filter = ActionFilterButter(sampling_rate=sampling_rate,
                                                    num_joints=self.num_motors)
        return a_filter

    def _ResetActionFilter(self):
        self._action_filter.reset()

    def _FilterAction(self, action):
        # initialize the filter history, since resetting the filter will fill
        # the history with zeros and this can cause sudden movements at the start
        # of each episode
        if self._step_counter == 0:
            default_action = self.GetMotorAngles()
            self._action_filter.init_history(default_action)

        filtered_action = self._action_filter.filter(action)
        return filtered_action

    def ApplyAction(self, motor_commands, idx=0):
        """Set the desired motor angles to the motors of the robot.

        The desired motor angles are clipped based on the maximum allowed velocity.
        If the pd_control_enabled is True, a torque is calculated according to
        the difference between current and desired joint angle, as well as the joint
        velocity. This torque is exerted to the motor. For more information about
        PD control, please refer to: https://en.wikipedia.org/wiki/PID_controller.

        Args:
          motor_commands: The desired motor angles.
          motor_kps: Proportional gains for the motor model. If not provided, it
            uses the default kp of the robot for all the motors.
          motor_kds: Derivative gains for the motor model. If not provided, it
            uses the default kd of the robot for all the motors.
        """

        if len(self._joint_max_velocity) > 0:
            current_motor_angle = self.GetTrueMotorAngles()
            max_velocities = np.asarray(self._joint_max_velocity)
            motor_commands_max = (current_motor_angle + self.time_step * max_velocities)
            motor_commands_min = (current_motor_angle - self.time_step * max_velocities)
            motor_commands = np.clip(motor_commands, motor_commands_min, motor_commands_max)
        
        # interpolates between the current and previous actions
        if self._last_action is not None:
            lerp = float(idx + 1) / self._action_repeat
            motor_commands = self._last_action + lerp * (motor_commands - self._last_action)

        # # Clipping motor commands to be clipped off based on the current motor angles
        # motor_commands = self._ClipMotorCommands(motor_commands)

        motor_commands_with_direction = np.multiply(motor_commands, self._motor_direction)
        self._last_action = motor_commands_with_direction # might come handy for action interpolation (smoothening the transitions)
        for motor_id, motor_command_with_direction, max_force in zip(self._motor_link_ids, motor_commands_with_direction, self._joint_max_force):
            self._SetDesiredMotorAngleById(motor_id, motor_command_with_direction, max_force)

    def Step(self, action):
        # A lowpass filter should be used to smooth actions.
        action = self._FilterAction(action)
        for i in range(self._action_repeat):
            self.ApplyAction(action, i)
            self._pybullet_client.stepSimulation()
            self.ReceiveObservation()
            self._step_counter += 1

    def ReceiveObservation(self):
        """Receive the observation from sensors.

        This function is called once per step. The observations are only updated
        when this function is called.
        """
        self._observation_history.appendleft(self.GetTrueObservation())
        self._control_observation = self._GetControlObservation()

    def GetObservation(self):
        observation = []
        observation.extend(self.GetMotorAngles()) # [0:12]
        observation.extend(self.GetMotorVelocities()) # [12:24]
        observation.extend(self.GetMotorTorques()) # [24:36]
        observation.extend(self.GetBaseRollPitchYaw()) # [36:39]
        observation.extend(self.GetBaseRollPitchYawRate()) # [39:42]
        self._last_velocity = self._current_velocity
        self._current_velocity = np.array(self.GetBaseVelocity())
        observation.extend(self._current_velocity - self._last_velocity) # [42:45]
        return observation

    def GetTrueObservation(self):
        observation = []
        observation.extend(self.GetTrueMotorAngles()) # [0:12]
        observation.extend(self.GetTrueMotorVelocities()) # [12:24]
        observation.extend(self.GetTrueMotorTorques()) # [24:36]
        observation.extend(self.GetTrueBaseOrientation()) # [36:40]
        observation.extend(self.GetTrueBaseRollPitchYawRate()) # [40:43]
        self._last_velocity = self._current_velocity
        self._current_velocity = np.array(self.GetBaseVelocity())
        observation.extend(self._current_velocity - self._last_velocity) # [43:46]
        return observation

    def GetBasePosition(self):
        """Get the position of robot's base.

        Returns:
          The position of robot's base.
        """
        position, _ = (self._pybullet_client.getBasePositionAndOrientation(self._robot_id))
        return position

    def GetBaseVelocity(self):
        """Get the linear velocity of robot's base.

        Returns:
        The velocity of robot's base.
        """
        velocity, _ = self._pybullet_client.getBaseVelocity(self._robot_id)
        return velocity

    def GetTrueBaseRollPitchYaw(self):
        """Get robot's base orientation in euler angle in the world frame.

        Returns:
          A tuple (roll, pitch, yaw) of the base in world frame.
        """
        orientation = self.GetTrueBaseOrientation()
        roll_pitch_yaw = self._pybullet_client.getEulerFromQuaternion(orientation)
        return np.asarray(roll_pitch_yaw)

    def GetBaseRollPitchYaw(self):
        """Get robot's base orientation in euler angle in the world frame.

        This function mimics the noisy sensor reading and adds latency.
        Returns:
          A tuple (roll, pitch, yaw) of the base in world frame polluted by noise
          and latency.
        """
        delayed_orientation = np.array(self._control_observation[3 * self.num_motors:3 * self.num_motors + 4])
        delayed_roll_pitch_yaw = self._pybullet_client.getEulerFromQuaternion(delayed_orientation)
        roll_pitch_yaw = self._AddSensorNoise(np.array(delayed_roll_pitch_yaw), self._observation_noise_stdev[3])
        return roll_pitch_yaw

    def GetTrueMotorAngles(self):
        """Gets the motor angles at the current moment, mapped to [-pi, pi].

        Returns:
          Motor angles, mapped to [-pi, pi].
        """
        motor_angles = [self._pybullet_client.getJointState(self._robot_id, motor_id)[0] for motor_id in self._motor_link_ids]
        motor_angles = np.multiply(motor_angles, self._motor_direction)
        return motor_angles

    def GetMotorAngles(self):
        """Gets the motor angles.

        This function mimicks the noisy sensor reading and adds latency. The motor
        angles that are delayed, noise polluted, and mapped to [-pi, pi].

        Returns:
          Motor angles polluted by noise and latency, mapped to [-pi, pi].
        """
        motor_angles = self._AddSensorNoise(
            np.array(self._control_observation[0:self.num_motors]), self._observation_noise_stdev[0])
        return MapToMinusPiToPi(motor_angles)

    def GetTrueMotorVelocities(self):
        """Get the velocity of all eight motors.

        Returns:
          Velocities of all eight motors.
        """
        motor_velocities = [self._pybullet_client.getJointState(self._robot_id, motor_id)[1] for motor_id in self._motor_link_ids]
        motor_velocities = np.multiply(motor_velocities, self._motor_direction)
        return motor_velocities

    def GetMotorVelocities(self):
        """Get the velocity of all eight motors.

        This function mimicks the noisy sensor reading and adds latency.
        Returns:
          Velocities of all eight motors polluted by noise and latency.
        """
        return self._AddSensorNoise(
            np.array(self._control_observation[self.num_motors:2 * self.num_motors]), self._observation_noise_stdev[1])

    def GetTrueMotorTorques(self):
        """Get the amount of torque the motors are exerting.

        Returns:
          Motor torques of all eight motors.
        """
        # if self._accurate_motor_model_enabled or self._pd_control_enabled:
        #     return self._observed_motor_torques
        # else:
        motor_torques = [self._pybullet_client.getJointState(self._robot_id, motor_id)[3] for motor_id in self._motor_link_ids]
        motor_torques = np.multiply(motor_torques, self._motor_direction)
        return motor_torques

    def GetMotorTorques(self):
        """Get the amount of torque the motors are exerting.

        This function mimicks the noisy sensor reading and adds latency.
        Returns:
          Motor torques of all eight motors polluted by noise and latency.
        """
        return self._AddSensorNoise(
            np.array(self._control_observation[2 * self.num_motors:3 * self.num_motors]), self._observation_noise_stdev[2])

    def GetTrueBaseOrientation(self):
        """Get the orientation of robot's base, represented as quaternion.

        Returns:
          The orientation of robot's base.
        """
        _, orientation = (self._pybullet_client.getBasePositionAndOrientation(self._robot_id))
        return orientation

    def GetBaseOrientation(self):
        """Get the orientation of robot's base, represented as quaternion.

        This function mimicks the noisy sensor reading and adds latency.
        Returns:
          The orientation of robot's base polluted by noise and latency.
        """
        return self._pybullet_client.getQuaternionFromEuler(self.GetBaseRollPitchYaw())

    def GetTrueBaseRollPitchYawRate(self):
        """Get the rate of orientation change of the robot's base in euler angle.

        Returns:
          rate of (roll, pitch, yaw) change of the robot's base.
        """
        vel = self._pybullet_client.getBaseVelocity(self._robot_id)
        return np.asarray([vel[1][0], vel[1][1], vel[1][2]])

    def GetBaseRollPitchYawRate(self):
        """Get the rate of orientation change of the robot's base in euler angle.

        This function mimicks the noisy sensor reading and adds latency.
        Returns:
          rate of (roll, pitch, yaw) change of the robot's base polluted by noise
          and latency.
        """
        return self._AddSensorNoise(
            np.array(self._control_observation[3 * self.num_motors + 4:3 * self.num_motors + 7]),
            self._observation_noise_stdev[4])

    def _SetDesiredMotorAngleById(self, motor_id, desired_angle, max_force):
        self._pybullet_client.setJointMotorControl2(bodyIndex=self._robot_id,
                                                    jointIndex=motor_id,
                                                    controlMode=self._pybullet_client.POSITION_CONTROL,
                                                    targetPosition=desired_angle)

    def GetJointLimits(self):
        """Gets the joint limits (angle, torque and velocity) of the robot"""
        
        return {
            "lower": np.array(self._joint_lower_limits, dtype=np.float32),
            "upper": np.array(self._joint_upper_limits, dtype=np.float32),
            "torque": np.array(self._joint_max_force, dtype=np.float32),
            "velocity": np.array(self._joint_max_velocity, dtype=np.float32)
        }

    def SetDesiredMotorAngleByName(self, motor_name, desired_angle):
        self._SetDesiredMotorAngleById(self._joint_name_to_id[motor_name], desired_angle)

    def _GetControlObservation(self):
        control_delayed_observation = self._GetDelayedObservation(self._control_latency)
        return control_delayed_observation

    def _GetDelayedObservation(self, latency):
        """Get observation that is delayed by the amount specified in latency.

        Args:
          latency: The latency (in seconds) of the delayed observation.
        Returns:
          observation: The observation which was actually latency seconds ago.
        """
        if latency <= 0 or len(self._observation_history) == 1:
            observation = self._observation_history[0]
        else:
            n_steps_ago = int(latency / self.time_step)
            if n_steps_ago + 1 >= len(self._observation_history):
                return self._observation_history[-1]
            remaining_latency = latency - n_steps_ago * self.time_step
            blend_alpha = remaining_latency / self.time_step
            observation = ((1.0 - blend_alpha) * np.array(self._observation_history[n_steps_ago]) +
                           blend_alpha * np.array(self._observation_history[n_steps_ago + 1]))
        return observation

    def _GetJointsInfo(self):
        """
        This function does 2 things:
        1) Creates a dictionary maps the joint name to its ID.
        2) Gets information about the limits applied on each
            joint based on the URDF file.

        Raises:
          ValueError: Unknown category of the joint name.
        """

        for i in range(self._num_joints):
            joint_info = self._pybullet_client.getJointInfo(self._robot_id, i)
            joint_id = joint_info[0]
            joint_name = joint_info[1].decode("UTF-8")
            self._joint_name_to_id[joint_name] = joint_id

            # Storing the ID each of each in a seperate array - might come handy later
            if HIP_NAME_PATTERN.match(joint_name):
                self._hip_link_ids.append(joint_id)
            elif UPPER_NAME_PATTERN.match(joint_name):
                self._upper_link_ids.append(joint_id)
            elif LOWER_NAME_PATTERN.match(joint_name):
                self._lower_link_ids.append(joint_id)
            elif TOE_NAME_PATTERN.match(joint_name):
                self._foot_link_ids.append(joint_id)
            elif IMU_NAME_PATTERN.match(joint_name):
                self._imu_link_ids.append(joint_id)
            else:
                raise ValueError("Unknown category of joint %s" % joint_name)

            # Getting lower limit, upper limit, max force and max velocity of each joint for building the actions the environment
            if joint_name in JOINT_NAMES:
                self._joint_lower_limits.append(joint_info[8])
                self._joint_upper_limits.append(joint_info[9])
                self._joint_max_force.append(joint_info[10])
                self._joint_max_velocity.append(joint_info[11])
                # All the motor Ids will be stored here
                self._motor_link_ids.append(joint_id)

        self._motor_link_ids.sort()
        self._hip_link_ids.sort()
        self._upper_link_ids.sort()
        self._lower_link_ids.sort()
        self._foot_link_ids.sort()

    def _AddSensorNoise(self, sensor_values, noise_stdev):
        if noise_stdev <= 0:
            return sensor_values
        observation = sensor_values + np.random.normal(scale=noise_stdev, size=sensor_values.shape)
        return observation

    def _ClipMotorCommands(self, motor_commands):
        """Clips motor commands.

        Args:
        motor_commands: np.array. Can be motor angles, torques, hybrid commands,
            or motor pwms (for robot only).

        Returns:
        Clipped motor commands.
        """

        # clamp the motor command by the joint limit, in case weired things happens
        max_angle_change = MAX_MOTOR_ANGLE_CHANGE_PER_STEP
        current_motor_angles = self.GetMotorAngles()
        motor_commands = np.clip(motor_commands,
                                current_motor_angles - max_angle_change,
                                current_motor_angles + max_angle_change)
        return motor_commands

    def _RemoveDefaultJointDamping(self):
        """Removes the damping on allthe joints"""
        for i in range(self._num_joints):
            joint_id = self._pybullet_client.getJointInfo(self._robot_id, i)[0]
            self._pybullet_client.changeDynamics(joint_id, -1, linearDamping=0, angularDamping=0)

    def SetTimeSteps(self, action_repeat, simulation_step):
        """Set the time steps of the control and simulation.

        Args:
          action_repeat: The number of simulation steps that the same action is
            repeated.
          simulation_step: The simulation time step.
        """
        self.time_step = simulation_step
        self._action_repeat = action_repeat
    
    def GetFootLinkIDs(self):
        """Get list of IDs for all foot links."""
        return self._foot_link_ids

    def GetTimeSinceReset(self):
        return self._step_counter * self.time_step

    @property
    def id(self):
        return self._robot_id

    @property
    def InitBasePosition(self):
        return self._init_base_position

    @property
    def InitBaseOrientation(self):
        return self._init_base_orientation

    @property
    def last_action(self):
        return self._last_action