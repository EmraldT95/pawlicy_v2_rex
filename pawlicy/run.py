import os
import inspect
import argparse

from pawlicy.envs import A1GymEnv
from pawlicy.learning import Trainer
from pawlicy.tasks import WalkAlongX

currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
SAVE_DIR = os.path.join(currentdir, "agents")


def main():
    # Getting all the arguments passed
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--mode', "-m", dest="mode", default="test", choices=[
                            "train", "test"], type=str, help='to set to training or testing mode')
    arg_parser.add_argument("--motor_control_mode", "-mcm", dest="motor_control_mode", default="Position",
                            choices=["Position", "Torque", "Velocity"], type=str, help="to set motor control mode")
    arg_parser.add_argument('--visualize', "-v", dest="visualize",
                            action="store_true", help='To flip rendering behaviour')
    arg_parser.add_argument("--randomise_terrain", "-rt", dest="randomise_terrain",
                            default=False, type=bool, help="to setup a randommized terrain")
    arg_parser.add_argument('--total_timesteps', "-tts", dest="total_timesteps",
                            default=int(1e6), type=int, help='total number of training steps')
    arg_parser.add_argument('--algorithm', "-a", dest="algorithm", default="SAC", choices=[
                            "SAC", "PPO", "TD3"], type=str, help='the algorithm used to train the robot')
    arg_parser.add_argument('--path', "-p", dest="path",
                            default='', type=str, help='the path to the saved model')
    arg_parser.add_argument('--max_episode_steps', "-mes", dest="max_episode_steps",
                            default=1000, type=int, help='maximum steps per episode')
    arg_parser.add_argument('--author', "-au", dest="author",
                            default="rpanackal", type=str, help='name of author')
    arg_parser.add_argument('--exp_suffix', "-s", dest="exp_suffix",
                            default="", type=str, help='appends to experiment name')
    arg_parser.add_argument('--total_num_eps', "-tne", dest="total_num_eps",
                            default=20, type=int, help='total number of test episodes')
    arg_parser.add_argument('--load_final_model', "-lfm", dest="load_final_model",
                            action="store_true", help='Whether to load the final model instead of best model')
    args = arg_parser.parse_args()

    task = WalkAlongX()

    # Setting the save path
    if args.path != '':
        path = os.path.join(currentdir, args.path)
    else:
        path = os.path.join(SAVE_DIR, args.algorithm)

    # Training
    if args.mode == "train":
        env = A1GymEnv(randomise_terrain=args.randomise_terrain,
                       motor_control_mode=args.motor_control_mode,
                       enable_rendering=args.visualize,
                       task=task)

        # Need to do this because our current pybullet setup can have only one client with GUI enabled
        eval_env = A1GymEnv(randomise_terrain=args.randomise_terrain,
                            motor_control_mode=args.motor_control_mode,
                            enable_rendering=False,
                            task=task)

        # Get the trainer
        local_trainer = Trainer(
            env, eval_env, args.algorithm, args.max_episode_steps, path)

        # The hyperparameters to override/add for the specific algorithm
        # (Check 'learning/hyperparams.yml' for default values)
        override_hyperparams = {
            "n_timesteps": args.total_timesteps,
            # "learning_rate_scheduler": "linear"
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
        Trainer(test_env, algorithm=args.algorithm,
                max_episode_steps=args.max_episode_steps, save_path=path).test()


if __name__ == "__main__":
    main()
