def launch_app():
    import argparse

    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description="BridgeDP IsaacSim RL framework")
    parser.add_argument("--debug", action="store_true")

    AppLauncher.add_app_launcher_args(parser)

    args_cli = parser.parse_args()
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    return args_cli, simulation_app


def main():
    import torch

    from isaaclab.envs import ManagerBasedRLEnv
    from tasks.T1 import T1ParkourDreamwaqCfg

    env = ManagerBasedRLEnv(cfg=T1ParkourDreamwaqCfg())

    env.reset()
    print("[INFO]: Setup complete...")
    # Run the simulator

    # Define simulation stepping
    count = 0
    # Simulation loop
    while simulation_app.is_running():
        # Perform step
        env.step(torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device))
        # Increment counter
        count += 1


if __name__ == '__main__':
    args_cli, simulation_app = launch_app()
    main()
    simulation_app.close()
