import os
import inspect
import argparse

from pawlicy.envs import A1GymEnv
from pawlicy.learning import Trainer
from pawlicy.tasks import WalkAlongX

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
SAVE_DIR = os.path.join(currentdir, "agents")

def main():
    # Getting all the arguments passed
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        '--mode', "-m",
        dest="mode",
        default="test",
        choices=["train", "test"],
        type=str,
        help='to set to training or testing mode')
    arg_parser.add_argument(
        "--motor_control_mode", "-mcm",
        dest="motor_control_mode",
        default="Position",
        choices=["Position", "Torque", "Velocity"],
        type=str,
        help="to set motor control mode")
    arg_parser.add_argument(
        '--visualize', "-v",
        dest="visualize",
        action="store_true",
        help='To flip rendering behaviour')
    arg_parser.add_argument(
        "--randomise_terrain", "-rt",
        dest="randomise_terrain",
        default=False, type=bool,
        help="to setup a randommized terrain")
    arg_parser.add_argument(
        '--total_timesteps', "-tts",
        dest="total_timesteps",
        default=int(1e6),
        type=int,
        help='total number of training steps')
    arg_parser.add_argument(
        '--algorithm', "-a",
        dest="algorithm",
        default="SAC",
        choices=["SAC", "PPO", "TD3"],
        type=str,
        help='the algorithm used to train the robot')
    args = arg_parser.parse_args()

    task = WalkAlongX()

    # Training
    if args.mode == "train":
        env = A1GymEnv(randomise_terrain=args.randomise_terrain,
                    motor_control_mode=args.motor_control_mode,
                    enable_rendering=args.visualize,
                    task=task)

        # Need to do this because our current pybullet setup can have only one client with GUI enabled
        if args.visualize:
            eval_env = None
        else:
            eval_env = A1GymEnv(randomise_terrain=args.randomise_terrain,
                        motor_control_mode=args.motor_control_mode,
                        enable_rendering=args.visualize,
                        task=task)

        # Get the trainer
        local_trainer = Trainer(env, eval_env, args.algorithm, max_episode_steps=500)

        # The hyperparameters to override/add for the specific algorithm
        # (Check 'learning/hyperparams.yml' for default values)
        override_hyperparams = {
            "n_timesteps": args.total_timesteps,
        }

        # Train the agent
        _ = local_trainer.train(override_hyperparams)

        # Save the model after training
        local_trainer.save_model()

    # Testing
    else:
        test_env = A1GymEnv(randomise_terrain=args.randomise_terrain,
                    motor_control_mode=args.motor_control_mode,
                    enable_rendering=True,
                    task=task)        
        Trainer(test_env,
                algorithm=args.algorithm,
                max_episode_steps=100000
                ).test(os.path.join(SAVE_DIR,
                                    args.algorithm))

if __name__ == "__main__":
    main()