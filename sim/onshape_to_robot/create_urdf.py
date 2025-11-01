from onshape_robotics_toolkit.connect import Client
from onshape_robotics_toolkit.log import LOGGER, LogLevel
from onshape_robotics_toolkit.robot import Robot
from argparse import ArgumentParser

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--name', default='robot')
    parser.add_argument('--url', required=True)
    args = parser.parse_args()

    LOGGER.set_file_name(f"{args.name}.log")
    LOGGER.set_stream_level(LogLevel.INFO)

    client = Client(env=".env")

    robot = Robot.from_url(
        name=args.name,
        url=args.url,
        client=client,
        max_depth=1,
        use_user_defined_root=False,
    )

    robot.save()
