import time
import gym
import numpy as np
import pybullet as p
import pybullet_data as pbd

from pybullet_utils.bullet_client import BulletClient
from gym.utils import seeding

from pawlicy.robot import A1
from pawlicy.envs import TerrainRandomizer
from pawlicy.tasks import DefaultTask

MOTOR_ANGLE_OBSERVATION_INDEX = 0
OBSERVATION_EPS = 0.01
RENDER_HEIGHT = 360
RENDER_WIDTH = 480
NUM_SIMULATION_ITERATION_STEPS = 300

class A1GymEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 60}

    def __init__(self,
                log_path = None,
                control_time_step=0.006,
                action_repeat=6,
                control_latency=0,
                enable_rendering=False,
                randomise_terrain=False,
                motor_control_mode="Position",
                task:DefaultTask=DefaultTask()):
        """Initializes the locomotion gym environment.

		Args:
            enable_rendering: Whether to run pybullet in GUI mode or not
            action_repeat: The number of simulation steps that the same action is repeated.
            randomise_terrain: Whether to randomize the terrains or not
            motor_control_mode: The mode in which the robot will operate. This will determine
                the action space.
			task: A callable function/class to calculate the reward and termination
				condition. Takes the gym env as the argument when calling.

		"""
        self._world_dict = {} # A dictionary containing the objects in the world other than the robot.
        self._task = task
        self._is_render = enable_rendering
        self._action_repeat = action_repeat
        self._control_latency = control_latency
        self._num_bullet_solver_iterations = int(NUM_SIMULATION_ITERATION_STEPS / self._action_repeat)
        self._randomise_terrain = randomise_terrain
        self._motor_control_mode = motor_control_mode
        self._log_path = log_path # Set up logging.
        self._cam_dist = 1.0
        self._cam_yaw = 0
        self._cam_pitch = -30

        if control_time_step is not None:
            self.control_time_step = control_time_step
            self._action_repeat = action_repeat
            self._time_step = control_time_step / action_repeat
        else:
            # Default values for time step and action repeat
            self._time_step = 0.01
            self._action_repeat = 1

        # Configure PyBullet
        if self._is_render:
            self._pb_client = BulletClient(connection_mode=p.GUI)
        else:
            self._pb_client = BulletClient()
        self._pb_client.setPhysicsEngineParameter(enableConeFriction=0)
        self._pb_client.setAdditionalSearchPath(pbd.getDataPath())

        self.seed()
        self.reset(True) # Hard reset initially to load the robot URDF file


    def reset(self, hard_reset=False, initial_motor_angles=None, reset_duration=0.5):
        """Resets the robot's position in the world or rebuild the sim world.

		The simulation world will be rebuilt if self._hard_reset is True.

		Args:
			initial_motor_angles: A list of Floats. The desired joint angles after
			    reset. If None, the robot will use its built-in value.
			reset_duration: Float. The time (in seconds) needed to rotate all motors
			    to the desired initial values.

		Returns:
			A numpy array contains the initial observation after reset.
		"""
        self._pb_client.configureDebugVisualizer(self._pb_client.COV_ENABLE_RENDERING, 0)

        # Clear the simulation world and reset the robot interface.
        if hard_reset:
            self._pb_client.resetSimulation()
            self._pb_client.setPhysicsEngineParameter(numSolverIterations=int(self._num_bullet_solver_iterations))
            self._pb_client.setTimeStep(self._time_step)
            self._pb_client.setGravity(0, 0, -9.8)

            # Generate the terrain
            terrain_id, terrain_type = self._generate_terrain(self._randomise_terrain)
            self._world_dict = { 
                "ground_id": terrain_id,
                "ground_type": terrain_type
            }

            # Build the robot
            self._robot = A1(
                    pybullet_client=self._pb_client,
                    action_repeat=self._action_repeat,
                    time_step=self._time_step,
                    # control_latency=self._control_latency,
                    terrain=terrain_type)

            # Create the action space and observation space
            self.action_space = self._build_action_space()
            self.observation_space = self._build_observation_space()

        # Reset the robot.
        self._robot.Reset(hard_reload=False, default_motor_angles=initial_motor_angles, reset_time=reset_duration)

        self._env_step_counter = 0
        self._env_time_step = self._time_step * self._action_repeat
        self._last_frame_time = 0.0 # The wall-clock time at which the last frame is rendered.
        if self._is_render:
            self._pb_client.configureDebugVisualizer(self._pb_client.COV_ENABLE_RENDERING, 1)
        self._pb_client.resetDebugVisualizerCamera(self._cam_dist, self._cam_yaw, self._cam_pitch, [0, 0, 0])
        self._last_action = np.zeros(self.action_space.shape)
        self._last_base_position = self._robot.InitBasePosition
        self._last_base_orientation = self._robot.InitBaseOrientation
        self._task.reset(self) # Reset the state of the task as well

        return self._get_observation()


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def step(self, action):
        """Step forward the simulation, given the action.

        Args:
            action: Can be a list of desired motor angles for all motors when the
                robot is in position control mode; A list of desired motor torques. Or a
                list of velocities for velocity control mode. The
                action must be compatible with the robot's motor control mode.

        Returns:
            observations: The observation based on current action
            reward: The reward for the current state-action pair.
            done: Whether the episode has ended.
            info: A dictionary that stores diagnostic information.

        Raises:
            ValueError: The action dimension is not the same as the number of motors.
            ValueError: The magnitude of actions is out of bounds.
        """
        self._last_base_position = self._robot.GetBasePosition()
        self._last_base_orientation = self._robot.GetBaseOrientation()
        self._last_action = action

        if self._is_render:
            # Sleep, otherwise the computation takes less time than real time,
            # which will make the visualization like a fast-forward video.
            # time_spent = time.time() - self._last_frame_time
            # self._last_frame_time = time.time()
            # time_to_sleep = self._env_time_step - time_spent
            # if time_to_sleep > 0:
            #     time.sleep(time_to_sleep)
            base_pos = self._robot.GetBasePosition()

            # Also keep the previous orientation of the camera set by the user.
            [yaw, pitch, dist] = self._pb_client.getDebugVisualizerCamera()[8:11]
            self._pb_client.resetDebugVisualizerCamera(dist, yaw, pitch, base_pos)
            # self._pb_client.configureDebugVisualizer(self._pb_client.COV_ENABLE_SINGLE_STEP_RENDERING, 1)

        self._act(action)
        observation = self._get_observation()
        reward = self._task.reward(self)
        done = self._termination()
        self._env_step_counter += 1
        return observation, reward, done, {}


    def render(self, mode='rgb_array'):
        """
        Renders the rgb view from the robots perspective.
        Currently tuned to get the view from the head of the robot
        """
        # Base information
        # proj_matrix = self._pb_client.computeProjectionMatrixFOV(fov=66, aspect=1, nearVal=0.01, farVal=100)
        # base_pos = list(self._robot.GetBasePosition())
        # base_ori = list(self._robot.GetTrueBaseOrientation())
        # base_pos[2] = base_pos[2]+0.1

        # # Rotate camera direction
        # rot_mat = np.array(self._pb_client.getMatrixFromQuaternion(base_ori)).reshape(3, 3)
        # camera_vec = np.matmul(rot_mat, [1, 0, 0]) # Target position is always the x-axis. This decides the diirection of the camera
        # up_vec = np.matmul(rot_mat, np.array([0, 0, 1])) # Which direction is considered "up"
        # view_matrix = self._pb_client.computeViewMatrix(base_pos, base_pos + camera_vec, up_vec)

        base_pos = self._robot.GetBasePosition()
        view_matrix = self._pb_client.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=base_pos,
            distance=self._cam_dist,
            yaw=self._cam_yaw,
            pitch=self._cam_pitch,
            roll=0,
            upAxisIndex=2)
        proj_matrix = self._pb_client.computeProjectionMatrixFOV(
            fov=60,
            aspect=float(RENDER_WIDTH) / RENDER_HEIGHT,
            nearVal=0.1,
            farVal=100.0)

        (_, _, px, _, _) = self._pb_client.getCameraImage(
            width=RENDER_WIDTH,
            height=RENDER_HEIGHT,
            renderer=self._pb_client.ER_BULLET_HARDWARE_OPENGL,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix)
        rgb_array = np.array(px)
        rgb_array = rgb_array[:, :, :3]
        return rgb_array


    def close(self):
        """Terminates the simulation"""
        self._pb_client.disconnect()
        self._robot.Terminate()

    def _termination(self):
        fallen = self.is_fallen()
        if fallen:
            print("FALLING DOWN!")
        return fallen or self._task.done(self)

    def is_fallen(self):
        """Decide whether robot has fallen.

        If the up directions between the base and the world is larger (the dot
        product is smaller than 0.85) or the base is very low on the ground
        (the height is smaller than 0.13 meter), robot is considered fallen.

        Returns:
          Boolean value that indicates whether robot has fallen.
        """
        orientation = self._robot.GetBaseOrientation()
        rot_mat = self._pb_client.getMatrixFromQuaternion(orientation)
        local_up = rot_mat[6:]
        return np.dot(np.asarray([0, 0, 1]), np.asarray(local_up)) < 0.85


    def _generate_terrain(self, randomize=False):
        """Generates terrain randomly (if needed)"""
        terrain_id = -1
        if randomize:
            terrain_id, terrain_type = TerrainRandomizer(self._pb_client).randomize()
        else:
            terrain_id = self._pb_client.loadURDF("plane.urdf")
            terrain_type = "plane"

        return terrain_id, terrain_type


    def _build_action_space(self):
        """Defines the action space of the gym environment"""
        # All limits defined according to urdf file
        joint_limits = self._robot.GetJointLimits()
        # Controls the torque applied at each motor
        if self._motor_control_mode == "Torque":
            high = joint_limits["torque"]
            action_space = gym.spaces.Box(-high, high, dtype=np.float32)
        # Controls the angles (in radians) of the joints
        elif self._motor_control_mode == "Position":
            action_space = gym.spaces.Box(joint_limits["lower"], joint_limits["upper"], dtype=np.float32)
        # Controls the velocity at which motors rotate
        elif self._motor_control_mode == "Velocity":
            high = joint_limits["velocity"]
            action_space = gym.spaces.Box(-high, high, dtype=np.float32)
        else:
            raise ValueError
        return action_space


    def _build_observation_space(self):
        """Defines the observation space of the gym environment"""
        joint_limits = self._robot.GetJointLimits()
        num_motors = self.robot.num_motors
        upper_bound = np.zeros(len(self._get_observation()))

        upper_bound[0:num_motors] = joint_limits["upper"]  # Joint angle.
        upper_bound[num_motors:2 * num_motors] = joint_limits["velocity"]  # Joint velocity.
        upper_bound[2 * num_motors:3 * num_motors] = joint_limits["torque"]  # Joint torque.
        upper_bound[3 * num_motors:] = 1.0  # Quaternion of base orientation.
        
        observation_space = gym.spaces.Box(-upper_bound, upper_bound, dtype=np.float64)
        return observation_space


    def _act(self, action):
        """Executes the action in the robot and also updates the task state"""
        self._robot.Step(action)
        self._task.update(self)


    def _get_observation(self):
        """Get observation of this environment, including noise and latency.

        robot class maintains a history of true observations. Based on the
        latency, this function will find the observation at the right time,
        interpolate if necessary. Then Gaussian noise is added to this observation
        based on self.observation_noise_stdev.

        Returns:
          The noisy observation with latency.
        """

        observation = []
        observation.extend(self._robot.GetMotorAngles().tolist())
        observation.extend(self._robot.GetMotorVelocities().tolist())
        observation.extend(self._robot.GetMotorTorques().tolist())
        observation.extend(list(self._robot.GetBaseOrientation()))
        self._observation = observation
        return self._observation

    def _get_true_observation(self):
        """Get the observations of this environment.

        It includes the angles, velocities, torques and the orientation of the base.

        Returns:
          The observation list. observation[0:8] are motor angles. observation[8:16]
          are motor velocities, observation[16:24] are motor torques.
          observation[24:28] is the orientation of the base, in quaternion form.
        """
        observation = []
        observation.extend(self._robot.GetTrueMotorAngles().tolist())
        observation.extend(self._robot.GetTrueMotorVelocities().tolist())
        observation.extend(self._robot.GetTrueMotorTorques().tolist())
        observation.extend(list(self._robot.GetTrueBaseOrientation()))

        self._true_observation = observation
        return self._true_observation


    def get_time_since_reset(self):
        """Get the time passed (in seconds) since the last reset.

        Returns:
            Time in seconds since the last reset.
        """
        return self._robot.GetTimeSinceReset()

    @property
    def pybullet_client(self):
        return self._pb_client

    @property
    def robot(self):
        return self._robot

    @property
    def last_action(self):
        return self._last_action

    @property
    def last_base_position(self):
        return self._last_base_position

    @property
    def env_step_counter(self):
        return self._env_step_counter

    @property
    def world_dict(self):
        return self._world_dict.copy()

    @world_dict.setter
    def world_dict(self, new_dict):
        self._world_dict = new_dict.copy()
