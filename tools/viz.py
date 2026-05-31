import os, sys
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("Loading data...", flush=True)
data_dir = Path("/workspace/starVLA/playground/data_downloads/libero/libero_spatial_no_noops_1.0.0_lerobot/data/chunk-000")
parquet_files = list(data_dir.glob("*.parquet"))
print(f"Found {len(parquet_files)} files", flush=True)

df = pd.read_parquet(parquet_files[0])
print(f"Data shape: {df.shape}", flush=True)

actions = np.array(df['action'].tolist())
print(f"Actions shape: {actions.shape}", flush=True)

output_dir = Path("/workspace/starVLA/outputs/visualization")
output_dir.mkdir(parents=True, exist_ok=True)

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
axes[0,0].plot(actions[:, :3])
axes[0,0].set_title('XYZ')
axes[0,1].plot(actions[:, 3:6])
axes[0,1].set_title('Rotation')
axes[1,0].plot(actions[:, 6], 'purple')
axes[1,0].set_title('Gripper')
axes[1,1].plot(actions[:, 0], actions[:, 1])
axes[1,1].set_title('XY Trajectory')

plt.tight_layout()
save_path = output_dir / "actions.png"
plt.savefig(save_path, dpi=150)
print(f"Saved: {save_path}", flush=True)

np.save(output_dir / "actions.npy", actions)
print("Done!", flush=True)
