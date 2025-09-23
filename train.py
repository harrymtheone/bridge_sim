def get_arg_parser():
    import argparse
    parser = argparse.ArgumentParser(description="BridgeDP IsaacSim RL framework")

    parser.add_argument('--proj_name', type=str, required=True)
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--exptid', type=str, required=True)
    parser.add_argument('--resume', action='store_true')
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
    import os
    from bridge_rl.runners import RLTaskCfg

    from tasks import all_tasks

    autodl_log_root = os.path.join(os.path.expanduser("~"), 'autodl-tmp')
    if os.path.isdir(autodl_log_root):
        log_root = os.path.join(autodl_log_root, 'logs')
    else:
        log_root = 'logs'

    task_cfg: RLTaskCfg = all_tasks[args.task]()
    task_cfg.log_root_dir = log_root
    task_cfg.project_name = args.proj_name
    task_cfg.exptid = args.exptid
    task_cfg.resume_id = getattr(args, 'resumeid', task_cfg.resume_id)
    task_cfg.checkpoint = getattr(args, 'checkpoint', task_cfg.checkpoint)

    if args.debug:
        task_cfg.env.scene.num_envs = 64
        task_cfg.logger_backend = None

    runner = task_cfg.class_type(cfg=task_cfg)

    runner.learn()


if __name__ == '__main__':
    args_cli, simulation_app = launch_app()
    main(args_cli)
    simulation_app.close()
