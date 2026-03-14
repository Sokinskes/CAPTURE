"""
Adaptive Horizon ACT 评估脚本
支持动态步长推理

使用方法:
python evaluate_adastep.py --task task1
"""

from config.config import POLICY_CONFIG, TASK_CONFIG, TRAIN_CONFIG, ROBOT_PORTS

import os
import cv2
import torch
import pickle
import argparse
from time import time
import numpy as np

from robot import Robot
from training.utils import *


# 解析命令行参数
parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='task1')
parser.add_argument('--ckpt', type=str, default='policy_best.ckpt', 
                   help='检查点名称')
args = parser.parse_args()
task = args.task

# 配置
cfg = TASK_CONFIG
policy_config = POLICY_CONFIG
train_cfg = TRAIN_CONFIG
device = os.environ['DEVICE']


def capture_image(cam):
    """捕获图像"""
    _, frame = cam.read()
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # 裁剪
    x1, y1 = 400, 0
    x2, y2 = 1600, 900
    image = image[y1:y2, x1:x2]
    # 调整大小
    image = cv2.resize(image, (cfg['cam_width'], cfg['cam_height']), 
                      interpolation=cv2.INTER_AREA)
    return image


if __name__ == "__main__":
    # 初始化相机
    cam = cv2.VideoCapture(cfg['camera_port'])
    if not cam.isOpened():
        raise IOError("无法打开摄像头")
    
    # 初始化机械臂
    follower = Robot(device_name=ROBOT_PORTS['follower'])

    # 加载策略
    checkpoint_dir = os.path.join(train_cfg['checkpoint_dir'], task)
    ckpt_path = os.path.join(checkpoint_dir, args.ckpt)
    
    print(f"\n{'='*60}")
    print(f"加载模型: {ckpt_path}")
    print(f"{'='*60}\n")
    
    policy = make_policy(policy_config['policy_class'], policy_config)
    loading_status = policy.load_state_dict(
        torch.load(ckpt_path, map_location=torch.device(device))
    )
    print(f"✓ 加载状态: {loading_status}")
    policy.to(device)
    policy.eval()

    # 加载数据统计
    stats_path = os.path.join(checkpoint_dir, f'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
    post_process = lambda a: a * stats['action_std'] + stats['action_mean']

    # 查询频率
    if policy_config['use_adastep']:
        print(f"✓ AdaStep 模式启用")
        print(f"  步长范围: [{policy_config['k_min']}, {policy_config['k_max']}]")
        use_adaptive = True
    else:
        print(f"✗ 固定步长模式")
        query_frequency = policy_config['num_queries']
        if policy_config['temporal_agg']:
            query_frequency = 1
            num_queries = policy_config['num_queries']
        use_adaptive = False

    # 预热
    print("\n🔧 预热中...")
    for i in range(90):
        follower.read_position()
        _ = capture_image(cam)
    
    # 初始化观测
    obs = {
        'qpos': pwm2pos(follower.read_position()),
        'qvel': vel2pwm(follower.read_velocity()),
        'images': {cn: capture_image(cam) for cn in cfg['camera_names']}
    }
    
    os.system('say "start"')
    print("\n" + "="*60)
    print("开始执行任务")
    print("="*60 + "\n")

    n_rollouts = 1
    for rollout_id in range(n_rollouts):
        print(f"📍 Rollout {rollout_id + 1}/{n_rollouts}")
        
        # 时序聚合缓冲区
        if policy_config['temporal_agg'] and not use_adaptive:
            all_time_actions = torch.zeros(
                [cfg['episode_len'], cfg['episode_len']+num_queries, cfg['state_dim']]
            ).to(device)
        
        qpos_history = torch.zeros((1, cfg['episode_len'], cfg['state_dim'])).to(device)
        
        # 用于记录和分析
        obs_replay = []
        action_replay = []
        horizon_history = []  # 记录每步的预测步长
        inference_times = []   # 记录推理时间
        
        with torch.inference_mode():
            t = 0
            action_chunk = None
            chunk_index = 0
            
            while t < cfg['episode_len']:
                # 预处理状态
                qpos_numpy = np.array(obs['qpos'])
                qpos = pre_process(qpos_numpy)
                qpos = torch.from_numpy(qpos).float().to(device).unsqueeze(0)
                qpos_history[:, t] = qpos
                curr_image = get_image(obs['images'], cfg['camera_names'], device)

                # 决策逻辑
                if use_adaptive:
                    # ===== AdaStep 模式 =====
                    if action_chunk is None or chunk_index >= len(action_chunk):
                        # 需要新的推理
                        start_time = time()
                        result = policy(qpos, curr_image)
                        
                        if isinstance(result, tuple):
                            all_actions, predicted_horizon = result
                            k = predicted_horizon.item()
                        else:
                            all_actions = result
                            k = policy_config['k_max']  # 默认值
                        
                        inference_time = (time() - start_time) * 1000  # ms
                        inference_times.append(inference_time)
                        
                        # 动态截断
                        action_chunk = all_actions[0, :k].cpu().numpy()
                        chunk_index = 0
                        horizon_history.append(k)
                        
                        print(f"  t={t:03d} | 预测步长: {k:2d} | "
                              f"推理耗时: {inference_time:.2f}ms")
                    
                    # 取当前动作
                    raw_action = action_chunk[chunk_index]
                    chunk_index += 1
                
                else:
                    # ===== 固定步长模式 =====
                    if t % query_frequency == 0:
                        start_time = time()
                        all_actions = policy(qpos, curr_image)
                        inference_time = (time() - start_time) * 1000
                        inference_times.append(inference_time)
                    
                    if policy_config['temporal_agg']:
                        all_time_actions[[t], t:t+num_queries] = all_actions
                        actions_for_curr_step = all_time_actions[:, t]
                        actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                        actions_for_curr_step = actions_for_curr_step[actions_populated]
                        k = 0.01
                        exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                        exp_weights = exp_weights / exp_weights.sum()
                        exp_weights = torch.from_numpy(exp_weights.astype(np.float32)).to(device).unsqueeze(dim=1)
                        raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                        raw_action = raw_action.squeeze(0).cpu().numpy()
                    else:
                        raw_action = all_actions[0, t % query_frequency].cpu().numpy()

                # 后处理动作
                action = post_process(raw_action)
                action = pos2pwm(action).astype(int)
                
                # 执行动作
                follower.set_goal_pos(action)

                # 更新观测
                obs = {
                    'qpos': pwm2pos(follower.read_position()),
                    'qvel': vel2pwm(follower.read_velocity()),
                    'images': {cn: capture_image(cam) for cn in cfg['camera_names']}
                }
                
                # 存储数据
                obs_replay.append(obs)
                action_replay.append(action)
                
                t += 1

        os.system('say "stop"')
        
        # ===== 统计分析 =====
        print("\n" + "="*60)
        print("执行完成 - 性能统计")
        print("="*60)
        
        if use_adaptive and horizon_history:
            print(f"\n📊 步长统计:")
            print(f"  平均步长: {np.mean(horizon_history):.2f}")
            print(f"  最小步长: {np.min(horizon_history)}")
            print(f"  最大步长: {np.max(horizon_history)}")
            print(f"  总推理次数: {len(horizon_history)}")
            
            # 计算节省的推理次数
            fixed_inferences = cfg['episode_len'] // policy_config['k_min']
            adaptive_inferences = len(horizon_history)
            saved = fixed_inferences - adaptive_inferences
            print(f"\n⚡ 效率提升:")
            print(f"  固定最小步长推理次数: {fixed_inferences}")
            print(f"  自适应推理次数: {adaptive_inferences}")
            print(f"  节省推理次数: {saved} ({saved/fixed_inferences*100:.1f}%)")
        
        if inference_times:
            print(f"\n⏱️  推理时间:")
            print(f"  平均: {np.mean(inference_times):.2f}ms")
            print(f"  最大: {np.max(inference_times):.2f}ms")
            print(f"  最小: {np.min(inference_times):.2f}ms")
        
        # 保存轨迹数据
        data_dict = {
            '/observations/qpos': [],
            '/observations/qvel': [],
            '/action': [],
        }
        for cam_name in cfg['camera_names']:
            data_dict[f'/observations/images/{cam_name}'] = []

        for o, a in zip(obs_replay, action_replay):
            data_dict['/observations/qpos'].append(o['qpos'])
            data_dict['/observations/qvel'].append(o['qvel'])
            data_dict['/action'].append(a)
            for cam_name in cfg['camera_names']:
                data_dict[f'/observations/images/{cam_name}'].append(o['images'][cam_name])

        # 保存到文件
        import h5py
        max_timesteps = len(data_dict['/observations/qpos'])
        data_dir = cfg['dataset_dir']
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        
        idx = len([name for name in os.listdir(data_dir) 
                  if os.path.isfile(os.path.join(data_dir, name))])
        save_path = os.path.join(data_dir, f'eval_episode_{idx}.hdf5')
        
        with h5py.File(save_path, 'w', rdcc_nbytes=1024**2*2) as root:
            root.attrs['sim'] = True
            obs = root.create_group('observations')
            image = obs.create_group('images')
            for cam_name in cfg['camera_names']:
                _ = image.create_dataset(
                    cam_name, 
                    (max_timesteps, cfg['cam_height'], cfg['cam_width'], 3), 
                    dtype='uint8',
                    chunks=(1, cfg['cam_height'], cfg['cam_width'], 3),
                )
            qpos = obs.create_dataset('qpos', (max_timesteps, cfg['state_dim']))
            qvel = obs.create_dataset('qvel', (max_timesteps, cfg['state_dim']))
            action = root.create_dataset('action', (max_timesteps, cfg['action_dim']))
            
            # 如果使用 AdaStep，保存步长历史
            if use_adaptive and horizon_history:
                horizon_ds = root.create_dataset('horizon_history', 
                                                 (len(horizon_history),), 
                                                 dtype='int32')
                horizon_ds[...] = np.array(horizon_history)
            
            for name, array in data_dict.items():
                root[name][...] = array
        
        print(f"\n✓ 轨迹已保存: {save_path}")
    
    # 关闭力矩
    follower._disable_torque()
    print("\n" + "="*60)
    print("评估完成")
    print("="*60 + "\n")
