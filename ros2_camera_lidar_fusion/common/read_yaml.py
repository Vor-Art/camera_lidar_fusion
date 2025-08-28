import yaml, os
from ament_index_python.packages import get_package_share_directory

def extract_configuration():
    config_file = os.path.join(
        get_package_share_directory('ros2_camera_lidar_fusion'),
        'config',
        'general_configuration.yaml'
    )

    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)

    return config