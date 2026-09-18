import os
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parent

# 构造两个脚本的绝对路径
cooling_scripts = root / "cooling_radius.py"
plot_scripts = root / "plotting_scripts" / "plot_cooling_radius_mass_trend.py"

snaps = [99, 67, 33]

data_dir = Path("/public/home/zju_visitor/LiuYuanhao/SAM_project/bymyself/data")

def main():
    env = os.environ.copy()

    old_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(root) + os.pathsep + old_pythonpath

    for snap in snaps:
        print(f"Running cooling calculation for snap {snap}")
        subprocess.run([sys.executable, str(cooling_scripts), str(snap) ], cwd=root, env=env, check=True)

    missing = [data_dir / f"cooling_radius_{snap}.hdf5" for snap in snaps if not (data_dir / f"cooling_radius_{snap}.hdf5").is_file()]
    if missing:
        raise FileNotFoundError(f"Missing cooling-radius files: {missing}")

    subprocess.run([sys.executable, str(plot_scripts)], cwd=root, env=env, check=True)

if __name__ == "__main__":
    main()