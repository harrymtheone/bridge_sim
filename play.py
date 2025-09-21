def get_arg_parser():
    import argparse
    parser = argparse.ArgumentParser(description="BridgeDP IsaacSim RL framework")

    parser.add_argument('--proj_name', type=str, required=True)
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--exptid', type=str, required=True)
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
    import vis
    import torch
    from isaaclab.terrains import TerrainImporterCfg
    from rich.live import Live
    from bridge_rl.runners import RLTaskCfg

    from tasks import all_tasks

    task_cfg: RLTaskCfg = all_tasks[args.task]()
    task_cfg.logger_backend = None
    task_cfg.log_root_dir = "logs"
    task_cfg.project_name = args.proj_name
    task_cfg.exptid = args.exptid
    task_cfg.resume_id = args.exptid
    task_cfg.checkpoint = getattr(args, "checkpoint", -1)

    task_cfg.env.scene.num_envs = 1
    if isinstance(task_cfg.env.scene.terrain, TerrainImporterCfg):
        task_cfg.env.scene.terrain.terrain_generator.num_rows = 4
        task_cfg.env.scene.terrain.terrain_generator.num_cols = 4

    runner = task_cfg.class_type(task_cfg)
    env = runner.env
    runner.algorithm.eval()

    observations, infos = env.reset()

    with Live(vis.gen_info_panel(env)) as live:
        while True:
            observations['use_estimated_values'] = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)  # TODO: not finished here?!

            actions = runner.algorithm.play_act(observations)

            # actions = {"joint_pos": env.motion_generator.get_motion('ref_motion') - env.scene['robot'].data.default_joint_pos}

            # actions = torch.zeros_like(actions)
            # phase = self.env.command_manager.default_term.get_phase()
            # actions[self.env.lookat_id, 0] = 0.1 * torch.sin(2 * torch.pi * phase[self.env.lookat_id, 0])

            # actions = {'action': actions}
            observations, rewards, terminated, timeouts, infos = env.step(actions)

            live.update(vis.gen_info_panel(env))

            # visualizer.plot(env)


if __name__ == '__main__':
    args_cli, simulation_app = launch_app()
    main(args_cli)
    simulation_app.close()
