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
    import vis
    import torch
    from isaaclab.terrains import TerrainImporterCfg
    from rich.live import Live

    from tasks import all_tasks
    args.resume = True

    task_cfg = all_tasks[args.task]()

    task_cfg.env_cfg.scene.num_envs = 2
    if isinstance(task_cfg.env_cfg.scene.ground, TerrainImporterCfg):
        task_cfg.env_cfg.scene.ground.terrain_generator.num_rows = 4
        task_cfg.env_cfg.scene.ground.terrain_generator.num_cols = 4

    task_cfg.env_cfg.sim.device = args.device
    task_cfg.logger_backend = None

    runner = task_cfg.class_type(task_cfg, args)
    env = runner.env
    runner.algorithm.eval()

    observations, infos = env.reset()

    with Live(vis.gen_info_panel(env)) as live:
        while True:
            observations['use_estimated_values'] = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)  # TODO: not finished here?!

            rtn = runner.algorithm.play_act(observations)
            actions = rtn['actions']

            # actions = {'action': rtn['actions'] * 0.}

            actions = env.motion_generator.get_motion('ref_motion') - env.scene['robot'].data.default_joint_pos

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
