from pawlicy.envs import A1GymEnv

def main():
    env = A1GymEnv(motor_control_mode="Position",
                    enable_rendering=True)

    while(1):
        try:
            obs, reward, done, info = env.step(env.action_space.sample())
            if done:
                env.reset()
            # print(reward)
        except ValueError:
            env.close()

if __name__ == "__main__":
    main()