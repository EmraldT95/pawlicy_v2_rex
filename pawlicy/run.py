from pawlicy.envs import A1GymEnv
from pawlicy.robot import A1

import pybullet as p
import pybullet_data as pbd
from pybullet_utils.bullet_client import BulletClient

# Some constants specific to the URDF file
LINK_NAME_ID_DICT = {
    "trunk" : -1,  "imu_link" : 0,
    "FR_hip_joint" : 1, "FR_upper_shoulder" : 2, "FR_upper_joint" : 3, "FR_lower_joint" : 4, "FR_toe" : 5,
    "FL_hip_joint" : 6, "FL_upper_shoulder" : 7, "FL_upper_joint" : 8, "FL_lower_joint" : 9, "FL_toe" : 10,
    "RR_hip_joint" : 11, "RR_upper_shoulder" : 12, "RR_upper_joint" : 13, "RR_lower_joint" : 14, "RR_toe" : 15,
    "RL_hip_joint" : 16, "RL_upper_shoulder" : 17, "RL_upper_joint" : 18, "RL_lower_joint" : 19, "RL_toe" : 20,
}
RANDOM_INIT_ANGLES = [
    -0.09224178, 3.6216, -2.6537266,0.27989328, 0.5252582, -2.202277, 0.52137035, 1.3827422, -1.7126653, 0.35403055, 1.9539814, -0.92294806]
JOINT_NAMES = [
    "FR_hip_joint", "FR_upper_joint", "FR_lower_joint",
    "FL_hip_joint", "FL_upper_joint", "FL_lower_joint",
    "RR_hip_joint", "RR_upper_joint", "RR_lower_joint",
    "RL_hip_joint", "RL_upper_joint", "RL_lower_joint",
]

def main():
    env = A1GymEnv(motor_control_mode="Position",
                    enable_rendering=True)

    # pb_client = BulletClient(connection_mode=p.GUI)
    # pb_client.setAdditionalSearchPath(pbd.getDataPath())
    # pb_client.setTimeStep(1/240)
    # pb_client.setGravity(0, 0, -9.8)

    # ground = pb_client.loadURDF("plane.urdf", [0, 0, 0])

    # robot = pb_client.loadURDF("a1/a1.urdf", [0, 0, 0.32], [0, 0, 0, 1])

    for _ in range(500):
        try:
            # for idx, name in enumerate(JOINT_NAMES):
            #     pb_client.setJointMotorControl2(bodyIndex=robot,
            #                                             jointIndex=LINK_NAME_ID_DICT[name],
            #                                             controlMode=pb_client.POSITION_CONTROL,
            #                                             targetPosition=RANDOM_INIT_ANGLES[idx],
            #                                             # positionGain=self._kp,
            #                                             # velocityGain=self._kd,
            #                                             force=3.5)
            # pb_client.stepSimulation()

            env.render()
            action = env.action_space.sample()
            print(action)
            obs, reward, done, info = env.step(action)
            if done:
                env.reset()
            print(reward)
        except ValueError:
            # pb_client.disconnect()
            env.close()

if __name__ == "__main__":
    main()