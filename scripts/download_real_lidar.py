"""Download authentic real-world automotive Velodyne LiDAR PCD frames."""

from pathlib import Path
import urllib.request

REPO_ROOT = Path(__file__).resolve().parent.parent
TARGET_DIR = REPO_ROOT / "data" / "samples" / "synthetic" / "real_world_highway"
TARGET_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://raw.githubusercontent.com/udacity/SFND_Lidar_Obstacle_Detection/master/src/sensors/data/pcd/data_1/"


def download_real_lidar_samples(num_frames: int = 10) -> None:
    print(f"Downloading {num_frames} real-world automotive Velodyne LiDAR PCD frames to {TARGET_DIR}...")
    for i in range(num_frames):
        filename = f"{i:010d}.pcd"
        target_file = TARGET_DIR / f"frame_{i:04d}.pcd"
        url = BASE_URL + filename
        if not target_file.exists() or target_file.stat().st_size == 0:
            print(f"Fetching {filename} -> {target_file.name}...")
            urllib.request.urlretrieve(url, str(target_file))
            print(f"  Saved {target_file.name} ({target_file.stat().st_size} bytes)")
        else:
            print(f"  {target_file.name} already exists ({target_file.stat().st_size} bytes)")
    print("Real-world LiDAR dataset download complete!")


if __name__ == "__main__":
    download_real_lidar_samples()
