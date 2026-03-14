#!/usr/bin/env python3
"""
Gen3 kinesthetic teaching data recorder (40 Hz) for ACT-style HDF5 episodes.
- observations/qpos: (T, joint_dim)
- observations/images/cam_high: (T, H, W, 3) RGB
- action: (T, joint_dim), defined as next-step target qpos

Replace the TODO sections with your Gen3/Kortex API calls.
"""

import time
import threading
from pathlib import Path
import h5py
import numpy as np

try: 
    import pyrealsense2 as rs
except Exception:  # noqa: BLE001
    rs = None


class DataRecorder:
    def __init__(self, hz=40, out_dir=None):
        self.hz = hz
        self.dt = 1.0 / hz
        self.out_dir = Path(out_dir or "../data").resolve()
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.qpos_list = []
        self.images_list = []

        self.pipeline = None
        self.latest_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.camera_running = False
        if rs is not None:
            self.pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)
            self.pipeline.start(config)
            self.camera_running = True
            self.camera_thread = threading.Thread(target=self._camera_update_loop, daemon=True)
            self.camera_thread.start()
            time.sleep(1.0)

        # TODO: initialize Gen3 arm and enable admittance/zero-g mode
        # self.arm = YourGen3API()
        # self.arm.enable_admittance_mode()

    def _camera_update_loop(self):
        while self.camera_running:
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=100)
                color_frame = frames.get_color_frame()
                if color_frame:
                    self.latest_frame = np.asanyarray(color_frame.get_data())
            except RuntimeError:
                continue

    def get_camera_frame(self):
        if self.pipeline is None:
            return np.zeros((480, 640, 3), dtype=np.uint8)
        return self.latest_frame.copy()

    def get_qpos(self):
        # TODO: return current joint angles (+ gripper) as 1D np.ndarray
        # return self.arm.get_joint_angles()
        return np.zeros(8, dtype=np.float32)

    def record_episode(self, episode_idx, max_time=10.0):
        print(f"准备录制 Episode {episode_idx}... 3 秒后开始")
        time.sleep(3)
        print("🔴 录制开始")

        self.qpos_list.clear()
        self.images_list.clear()

        max_steps = int(max_time * self.hz)
        next_tick = time.perf_counter()
        drop_count = 0
        for _ in range(max_steps):

            qpos = self.get_qpos()
            img = self.get_camera_frame()

            self.qpos_list.append(qpos)
            self.images_list.append(img)

            next_tick += self.dt
            sleep_time = next_tick - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                drop_count += 1
                if sleep_time < -self.dt * 2:
                    next_tick = time.perf_counter()

        if drop_count > 0:
            print(f"⚠️ 本次录制轻微掉帧: {drop_count}/{max_steps} 步延迟超限")

        print("⏹️ 录制结束, 写入HDF5")
        out_path = self.out_dir / f"episode_{episode_idx}.hdf5"
        self.save_to_hdf5(out_path)
        print(f"✅ 保存到: {out_path}")

    def __del__(self):
        self.camera_running = False
        if self.pipeline is not None:
            self.pipeline.stop()

    def save_to_hdf5(self, filepath: Path):
        qpos_arr = np.asarray(self.qpos_list, dtype=np.float32)
        images_arr = np.asarray(self.images_list, dtype=np.uint8)

        action_arr = np.copy(qpos_arr)
        if len(action_arr) > 1:
            action_arr[:-1] = qpos_arr[1:]
            action_arr[-1] = qpos_arr[-1]

        with h5py.File(filepath, "w") as root:
            obs = root.create_group("observations")
            obs.create_dataset("qpos", data=qpos_arr, compression="gzip")

            imgs = obs.create_group("images")
            images_rgb = images_arr[..., ::-1]
            imgs.create_dataset("cam_high", data=images_rgb, dtype="uint8", chunks=True)

            root.create_dataset("action", data=action_arr, compression="gzip")


if __name__ == "__main__":
    recorder = DataRecorder(hz=50, out_dir=Path(__file__).parent.parent / "data")
    for i in range(5):
        recorder.record_episode(episode_idx=i, max_time=10.0)
        input("按 Enter 键开始录制下一条...")
