import os
import h5py
import torch
import numpy as np
from einops import rearrange
from torch.utils.data import DataLoader

from training.policy import ACTPolicy, CNNMLPPolicy


import IPython
e = IPython.embed

class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, episode_ids, dataset_dir, camera_names, norm_stats, use_adastep=False, horizon_labels_dict=None, episode_files=None):
        super(EpisodicDataset).__init__()
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.is_sim = None
        self.use_adastep = use_adastep
        self.horizon_labels_dict = horizon_labels_dict if horizon_labels_dict is not None else {}
        self.episode_files = episode_files if episode_files is not None else [f'episode_{i}.hdf5' for i in range(len(episode_ids))]
        #self.__getitem__(0) # initialize self.is_sim

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        sample_full_episode = False # hardcode

        episode_id = self.episode_ids[index]
        dataset_path = os.path.join(self.dataset_dir, self.episode_files[episode_id])
        with h5py.File(dataset_path, 'r') as root:
            is_sim = root.attrs['sim']
            original_action_shape = root['/action'].shape
            episode_len = original_action_shape[0]
            if sample_full_episode:
                start_ts = 0
            else:
                start_ts = np.random.choice(episode_len)
            # get observation at start_ts only
            qpos = root['/observations/qpos'][start_ts]
            qvel = root['/observations/qvel'][start_ts]
            image_dict = dict()
            for cam_name in self.camera_names:
                image_dict[cam_name] = root[f'/observations/images/{cam_name}'][start_ts]
            # get all actions after and including start_ts
            if is_sim:
                action = root['/action'][start_ts:]
                action_len = episode_len - start_ts
            else:
                action = root['/action'][max(0, start_ts - 1):] # hack, to make timesteps more aligned
                action_len = episode_len - max(0, start_ts - 1) # hack, to make timesteps more aligned

        self.is_sim = is_sim
        padded_action = np.zeros(original_action_shape, dtype=np.float32)
        padded_action[:action_len] = action
        is_pad = np.zeros(episode_len)
        is_pad[action_len:] = 1

        # new axis for different cameras
        all_cam_images = []
        for cam_name in self.camera_names:
            all_cam_images.append(image_dict[cam_name])
        all_cam_images = np.stack(all_cam_images, axis=0)

        # construct observations
        image_data = torch.from_numpy(all_cam_images)
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        # channel last
        image_data = torch.einsum('k h w c -> k c h w', image_data)

        # normalize image and change dtype to float
        image_data = image_data / 255.0
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]

        padded_horizon_labels = np.zeros(episode_len, dtype=np.float32)
        if episode_id in self.horizon_labels_dict:
            labels = self.horizon_labels_dict[episode_id]
            padded_horizon_labels[:len(labels)] = labels

        if self.use_adastep:
            horizon_labels = torch.from_numpy(padded_horizon_labels).float()
            return image_data, qpos_data, action_data, is_pad, horizon_labels
        else:
            return image_data, qpos_data, action_data, is_pad


def get_norm_stats(dataset_dir, num_episodes):
    episode_files = [f for f in os.listdir(dataset_dir) if f.endswith('.hdf5')]
    episode_files = sorted(episode_files)[:num_episodes]  # take first num_episodes
    all_qpos_data = []
    all_action_data = []
    for episode_file in episode_files:
        dataset_path = os.path.join(dataset_dir, episode_file)
        with h5py.File(dataset_path, 'r') as root:
            qpos = root['/observations/qpos'][()]
            qvel = root['/observations/qvel'][()]
            action = root['/action'][()]
        all_qpos_data.append(torch.from_numpy(qpos))
        all_action_data.append(torch.from_numpy(action))
    # concatenate instead of stack
    all_qpos_data = torch.cat(all_qpos_data, dim=0)
    all_action_data = torch.cat(all_action_data, dim=0)

    # normalize action data
    action_mean = all_action_data.mean(dim=[0], keepdim=True)
    action_std = all_action_data.std(dim=[0], keepdim=True)
    action_std = torch.clip(action_std, 1e-2, np.inf) # clipping

    # normalize qpos data
    qpos_mean = all_qpos_data.mean(dim=[0], keepdim=True)
    qpos_std = all_qpos_data.std(dim=[0], keepdim=True)
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf) # clipping

    stats = {"action_mean": action_mean.numpy().squeeze(), "action_std": action_std.numpy().squeeze(),
             "qpos_mean": qpos_mean.numpy().squeeze(), "qpos_std": qpos_std.numpy().squeeze(),
             "example_qpos": qpos}

    return stats


def load_data(dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val, policy_config=None):
    print(f'\nData from: {dataset_dir}\n')
    use_adastep = policy_config.get('use_adastep', False) if policy_config else False
    horizon_labels_dict = {}
    episode_files = [f for f in os.listdir(dataset_dir) if f.endswith('.hdf5')]
    episode_files = sorted(episode_files)[:num_episodes]
    if use_adastep:
        print('Generating AdaStep horizon labels...')
        from training.adastep import StateClusterAnalyzer
        analyzer = StateClusterAnalyzer(
            num_clusters=policy_config.get('num_clusters', 10),
            error_threshold=policy_config.get('error_threshold', 0.5)
        )
        # collect all states and actions
        all_states = []
        all_action_seqs = []
        for episode_file in episode_files:
            dataset_path = os.path.join(dataset_dir, episode_file)
            with h5py.File(dataset_path, 'r') as root:
                qpos = root['/observations/qpos'][()]
                action = root['/action'][()]
            all_states.append(qpos)
            all_action_seqs.append(action)
        concatenated_states = np.concatenate(all_states, axis=0)
        analyzer.fit_clusters(concatenated_states)
        analyzer.pareto_analysis(all_states, all_action_seqs, 
                                k_min=policy_config.get('k_min', 5), 
                                k_max=policy_config.get('k_max', 50))
        # generate labels for each episode
        for idx, episode_file in enumerate(episode_files):
            dataset_path = os.path.join(dataset_dir, episode_file)
            with h5py.File(dataset_path, 'r') as root:
                qpos = root['/observations/qpos'][()]
            labels = analyzer.get_labels(qpos)
            horizon_labels_dict[idx] = labels.squeeze()
        print('AdaStep labels generated.')
    
    # obtain train test split
    shuffled_indices = np.random.permutation(num_episodes)
    train_indices = shuffled_indices[:int(0.8 * num_episodes)]
    val_indices = shuffled_indices[int(0.8 * num_episodes):]

    # obtain normalization stats for qpos and action
    norm_stats = get_norm_stats(dataset_dir, num_episodes)

    # construct dataset and dataloader
    train_dataset = EpisodicDataset(train_indices, dataset_dir, camera_names, norm_stats, use_adastep, horizon_labels_dict, episode_files)
    val_dataset = EpisodicDataset(val_indices, dataset_dir, camera_names, norm_stats, use_adastep, horizon_labels_dict, episode_files)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)

    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim

def make_policy(policy_class, policy_config):
    if policy_class == "ACT":
        policy = ACTPolicy(policy_config)
    elif policy_class == "CNNMLP":
        policy = CNNMLPPolicy(policy_config)
    else:
        raise ValueError(f"Unknown policy class: {policy_class}")
    return policy

def make_optimizer(policy_class, policy):
    if policy_class == 'ACT':
        optimizer = policy.configure_optimizers()
    elif policy_class == 'CNNMLP':
        optimizer = policy.configure_optimizers()
    else:
        raise ValueError(f"Unknown policy class: {policy_class}")
    return optimizer

### env utils

def sample_box_pose():
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])

def sample_insertion_pose():
    # Peg
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

    # Socket
    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])

    return peg_pose, socket_pose

### helper functions

def get_image(images, camera_names, device='cpu'):
    curr_images = []
    for cam_name in camera_names:
        curr_image = rearrange(images[cam_name], 'h w c -> c h w')
        curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().to(device).unsqueeze(0)
    return curr_image

def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result

def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def pos2pwm(pos:np.ndarray) -> np.ndarray:
    """
    :param pos: numpy array of joint positions in range [-pi, pi]
    :return: numpy array of pwm values in range [0, 4096]
    """ 
    return (pos / 3.14 + 1.) * 2048
    
def pwm2pos(pwm:np.ndarray) -> np.ndarray:
    """
    :param pwm: numpy array of pwm values in range [0, 4096]
    :return: numpy array of joint positions in range [-pi, pi]
    """
    return (pwm / 2048 - 1) * 3.14

def pwm2vel(pwm:np.ndarray) -> np.ndarray:
    """
    :param pwm: numpy array of pwm/s joint velocities
    :return: numpy array of rad/s joint velocities 
    """
    return pwm * 3.14 / 2048

def vel2pwm(vel:np.ndarray) -> np.ndarray:
    """
    :param vel: numpy array of rad/s joint velocities
    :return: numpy array of pwm/s joint velocities
    """
    return vel * 2048 / 3.14
    
def pwm2norm(x:np.ndarray) -> np.ndarray:
    """
    :param x: numpy array of pwm values in range [0, 4096]
    :return: numpy array of values in range [0, 1]
    """
    return x / 4096
    
def norm2pwm(x:np.ndarray) -> np.ndarray:
    """
    :param x: numpy array of values in range [0, 1]
    :return: numpy array of pwm values in range [0, 4096]
    """
    return x * 4096