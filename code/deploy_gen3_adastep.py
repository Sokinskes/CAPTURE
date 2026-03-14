#!/usr/bin/env python3
"""
AdaStep real-robot deployment skeleton for Kinova Gen3.
- Reads camera + qpos
- Runs ACT + AdaStep predictor
- Executes open-loop chunk at 50 Hz
- Logs k_t and inference latency

Fill in TODOs with your Kortex API and model loading code.
"""

import time
import os
import pickle
from pathlib import Path
import threading
import numpy as np

try:
    import rospy
    from std_msgs.msg import Empty
    from sensor_msgs.msg import JointState
    from kortex_driver.msg import Base_JointSpeeds, JointSpeed, GripperCommand, Finger, GripperMode
    from kortex_driver.srv import SendGripperCommand
except Exception:  # noqa: BLE001
    rospy = None

try:
    import torch
except Exception:  # noqa: BLE001
    torch = None

try:
    import pyrealsense2 as rs
except Exception:  # noqa: BLE001
    rs = None

try:
    from config.config import POLICY_CONFIG, TRAIN_CONFIG, TASK_CONFIG
    from training.utils import make_policy, get_image
except Exception:  # noqa: BLE001
    POLICY_CONFIG = None
    TRAIN_CONFIG = None
    TASK_CONFIG = None
    make_policy = None
    get_image = None


class RealSenseThread:
    def __init__(self, width=640, height=480, fps=60):
        self.latest_frame = np.zeros((height, width, 3), dtype=np.uint8)
        self.running = False
        self.pipeline = None

        if rs is not None:
            self.pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
            self.pipeline.start(config)
            self.running = True
            self.thread = threading.Thread(target=self._loop, daemon=True)
            self.thread.start()
            time.sleep(1.0)

    def _loop(self):
        while self.running:
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=100)
                color_frame = frames.get_color_frame()
                if color_frame:
                    self.latest_frame = np.asanyarray(color_frame.get_data())
            except RuntimeError:
                continue

    def get_latest_frame(self):
        return self.latest_frame.copy()

    def close(self):
        self.running = False
        if self.pipeline is not None:
            self.pipeline.stop()


class KinovaController:
    def __init__(self, robot_name="my_gen3", hz=40, max_joint_speed=1.0, gripper_deadband=0.05):
        self.hz = hz
        self.dt = 1.0 / hz
        self.robot_name = robot_name
        self.max_joint_speed = max_joint_speed
        self.gripper_deadband = gripper_deadband
        self._latest_joint_state = None
        self.last_gripper_cmd = None

        if rospy is None:
            raise RuntimeError("rospy is required for ROS control")

        self.joint_state_sub = rospy.Subscriber(
            f"/{self.robot_name}/joint_state", JointState, self._joint_state_cb, queue_size=1
        )
        self.joint_vel_pub = rospy.Publisher(
            f"/{self.robot_name}/in/joint_velocity", Base_JointSpeeds, queue_size=1
        )
        self.stop_pub = rospy.Publisher(
            f"/{self.robot_name}/in/stop", Empty, queue_size=1
        )

        gripper_srv = f"/{self.robot_name}/base/send_gripper_command"
        rospy.wait_for_service(gripper_srv)
        self.send_gripper_command_srv = rospy.ServiceProxy(gripper_srv, SendGripperCommand)

    def _joint_state_cb(self, msg: JointState):
        self._latest_joint_state = msg

    def get_qpos(self):
        if self._latest_joint_state is None:
            return np.zeros(8, dtype=np.float32)
        pos = np.asarray(self._latest_joint_state.position, dtype=np.float32)
        if pos.size >= 8:
            return pos[:8]
        qpos = np.zeros(8, dtype=np.float32)
        qpos[:pos.size] = pos
        return qpos

    def send_joint_commands(self, target_joints):
        current = self.get_qpos()[:7]
        target = np.asarray(target_joints, dtype=np.float32)
        if target.size != 7:
            return
        vel = (target - current) * self.hz
        vel = np.clip(vel, -self.max_joint_speed, self.max_joint_speed)

        msg = Base_JointSpeeds()
        msg.duration = 0
        msg.joint_speeds = []
        for j in range(7):
            js = JointSpeed()
            js.joint_identifier = j
            js.value = float(vel[j])
            js.duration = 0
            msg.joint_speeds.append(js)
        self.joint_vel_pub.publish(msg)

    def send_gripper_command(self, target_gripper):
        if target_gripper is None or np.isnan(target_gripper):
            return
        if self.last_gripper_cmd is not None:
            if abs(float(target_gripper) - float(self.last_gripper_cmd)) < self.gripper_deadband:
                return
        self.last_gripper_cmd = float(target_gripper)
        cmd = GripperCommand()
        cmd.mode = GripperMode.GRIPPER_POSITION
        finger = Finger()
        finger.finger_identifier = 0
        finger.value = float(target_gripper)
        cmd.gripper.finger = [finger]
        cmd.duration = 0
        try:
            self.send_gripper_command_srv(cmd)
        except Exception:
            return

    def stop(self):
        self.stop_pub.publish(Empty())


def preprocess_image(img_bgr, camera_names):
    if torch is None or get_image is None:
        raise RuntimeError("torch/training.utils not available")
    images = {camera_names[0]: img_bgr[..., ::-1].copy()}
    return get_image(images, camera_names, device="cuda")


def denormalize_and_round_k(k_norm, k_min=5, k_max=50):
    k = k_norm * (k_max - k_min) + k_min
    return int(np.clip(np.round(k), k_min, k_max))


def run_adastep_deployment():
    if rospy is None:
        raise RuntimeError("rospy not available")
    rospy.init_node("adastep_deploy")

    camera = RealSenseThread()
    robot_name = rospy.get_param("~robot_name", "my_gen3")
    max_joint_speed = rospy.get_param("~max_joint_speed", 0.2)
    gripper_deadband = rospy.get_param("~gripper_deadband", 0.05)
    robot = KinovaController(
        robot_name=robot_name,
        hz=40,
        max_joint_speed=max_joint_speed,
        gripper_deadband=gripper_deadband,
    )

    if POLICY_CONFIG is None or TRAIN_CONFIG is None or TASK_CONFIG is None:
        raise RuntimeError("ACT config not available")

    task = rospy.get_param("~task", "task1")
    ckpt_name = rospy.get_param("~ckpt", "policy_last.ckpt")
    stats_path = rospy.get_param("~stats", "dataset_stats.pkl")

    policy_config = POLICY_CONFIG.copy()
    policy_config["use_adastep"] = rospy.get_param("~use_adastep", True)
    policy_config["k_min"] = rospy.get_param("~k_min", policy_config.get("k_min", 5))
    policy_config["k_max"] = rospy.get_param("~k_max", policy_config.get("k_max", 50))

    # Load ACT(+AdaStep) policy
    ckpt_dir = os.path.join(TRAIN_CONFIG["checkpoint_dir"], task)
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    policy = make_policy(policy_config["policy_class"], policy_config)
    loading_status = policy.load_state_dict(torch.load(ckpt_path, map_location="cuda"))
    rospy.loginfo(loading_status)
    policy.to("cuda").eval()
    rospy.loginfo(f"Loaded policy: {ckpt_path}")

    # Load dataset stats (qpos/action normalization)
    stats_file = stats_path if os.path.isabs(stats_path) else os.path.join(ckpt_dir, stats_path)
    with open(stats_file, "rb") as f:
        stats = pickle.load(f)
    qpos_mean = torch.tensor(stats["qpos_mean"], dtype=torch.float32, device="cuda").unsqueeze(0)
    qpos_std = torch.tensor(stats["qpos_std"], dtype=torch.float32, device="cuda").unsqueeze(0)
    action_mean = torch.tensor(stats["action_mean"], dtype=torch.float32, device="cuda").unsqueeze(0)
    action_std = torch.tensor(stats["action_std"], dtype=torch.float32, device="cuda").unsqueeze(0)

    camera_names = policy_config.get("camera_names", ["front"])

    k_log = []
    latency_log = []

    try:
        print("🚀 开始闭环控制")
        while not rospy.is_shutdown():
            t_start = time.perf_counter()

            img = camera.get_latest_frame()
            qpos = robot.get_qpos()

            img_tensor = preprocess_image(img, camera_names)
            qpos_tensor = torch.tensor(qpos, dtype=torch.float32, device="cuda").unsqueeze(0)
            qpos_tensor_norm = (qpos_tensor - qpos_mean) / qpos_std

            result = policy(qpos_tensor_norm, img_tensor)
            if isinstance(result, tuple):
                all_actions_norm, predicted_horizon = result
                k_t = int(predicted_horizon.item())
            else:
                all_actions_norm = result
                k_t = int(policy_config.get("k_max", 50))

            action_chunk_norm = all_actions_norm[0, :k_t]
            action_chunk = (action_chunk_norm * action_std + action_mean).detach().cpu().numpy()

            infer_latency = time.perf_counter() - t_start
            k_log.append(k_t)
            latency_log.append(infer_latency)

            for step_idx in range(k_t):
                step_start = time.perf_counter()
                target_action = action_chunk[step_idx]
                robot.send_joint_commands(target_action[:7])
                robot.send_gripper_command(target_action[7])
                sleep_time = robot.dt - (time.perf_counter() - step_start)
                if sleep_time > 0:
                    time.sleep(sleep_time)

    finally:
        robot.stop()
        camera.close()
        out_dir = Path(__file__).parent.parent / "data"
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "k_log.npy", np.asarray(k_log))
        np.save(out_dir / "latency_log.npy", np.asarray(latency_log))


if __name__ == "__main__":
    run_adastep_deployment()
