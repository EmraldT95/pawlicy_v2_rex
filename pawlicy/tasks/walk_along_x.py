import numpy as np

"""
This task to be used after setting action limits to 0.5 in all 3 motors of a leg
"""

class WalkAlongX(object):
    """Task to walk along a straight line (x-axis)"""
    def __init__(self,
                step_weight : float = 0.223,
                displacement_weight : float = 800,
                drift_weight: float = 16,
                orientation_weight : float = 1,
                action_cost_weight: float = 0.0004,
                ):
        """Initializes the task."""

        self._action_cost_weight = action_cost_weight
        self._drift_weight = drift_weight
        self._step_weight = step_weight
        self._orientation_weight = orientation_weight
        self._displacement_weight = displacement_weight

        self._target_pos =  np.array([100, 0, 1])

    def __call__(self, env):
        return self.reward(env)

    def reset(self, env):
        """Resets the internal state of the task."""
        self._env = env
        self._init_base_pos = np.array(env.robot.GetBasePosition())
        
        self._current_base_pos = self._init_base_pos
        self._last_base_pos = self._current_base_pos
        
        self._current_base_vel = env.robot.GetBaseVelocity()

        self._last_action = env.robot.last_action
        
        self._current_base_ori = env.robot.GetBaseOrientation()
        self._rot_mat = env.pybullet_client.getMatrixFromQuaternion(self._current_base_ori)
        
    def update(self, env):
        """Updates the internal state of the task.
        Evoked after call to a1.A1.Step(), ie after action takes effect in simulation
        """
        self._last_base_pos = self._current_base_pos
        self._current_base_pos = np.array(env.robot.GetBasePosition())

        self._current_base_vel = env.robot.GetBaseVelocity()
        self._last_action = env.last_action
        
        self._current_base_ori = env.robot.GetBaseOrientation()
        self._rot_mat = env.pybullet_client.getMatrixFromQuaternion(self._current_base_ori)


    def done(self, env):
        """Checks if the episode is over.

            If the robot base becomes unstable (based on orientation), the episode
            terminates early.
        """
        #rot_quat = env.robot.GetBaseOrientation()
        #rot_mat = env.pybullet_client.getMatrixFromQuaternion(self._current_base_ori)
        return self._rot_mat[-1] < 0.85


    def reward(self, env):
        
        displacement_reward = self._displacement_weight * (self._current_base_pos[0] - self._last_base_pos[0])

        action_reward = - self._action_cost_weight * np.linalg.norm(self._last_action)

        drift_reward =  - self._drift_weight * (self._current_base_pos[1]) ** 2
      
        orientation_reward = self._orientation_weight * self._rot_mat[-1] ** 2
        #print(f"displacement {displacement_reward} action {action_reward} orientation {orientation_reward}")
        reward = displacement_reward + action_reward + orientation_reward # + self._step_weight + drift_reward 

        return reward