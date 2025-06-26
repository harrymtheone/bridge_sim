def get_arg_parser():
    import argparse
    parser = argparse.ArgumentParser(description="BridgeDP IsaacSim RL framework")

    parser.add_argument('--proj_name', type=str, required=True)
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--exptid', type=str, required=True)
    parser.add_argument('--resumeid', type=str)
    parser.add_argument('--checkpoint', type=str)

    parser.add_argument('--log_root', type=str, default='logs')
    parser.add_argument('--debug', action='store_true')
    return parser


def launch_app():
    from isaaclab.app import AppLauncher

    parser = get_arg_parser()
    AppLauncher.add_app_launcher_args(parser)

    args = parser.parse_args()
    app_launcher = AppLauncher(args)
    sim_app = app_launcher.app
    return args, sim_app


def main(args):
    from bridge_rl.runners import RLRunner, RLRunnerCfg
    from tasks.T1 import T1ParkourDreamwaqCfg
    from bridge_rl.algorithms.dreamwaq import DreamWaQCfg

    runner = RLRunner(
        cfg=RLRunnerCfg(
            task_cfg=T1ParkourDreamwaqCfg(),
            algorithm_cfg=DreamWaQCfg(),
            max_iterations=1000,
        ),
        args=args,
    )

    runner.learn()


if __name__ == '__main__':
    args_cli, simulation_app = launch_app()
    main(args_cli)
    simulation_app.close()
