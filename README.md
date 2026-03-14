# CAPTURE

Minimal training + real-robot data collection/deployment bundle.

## Contents
- train.py / evaluate_adastep.py
- training/ (policy + utils)
- config/
- detr/
- code/ (record_gen3_hdf5.py, deploy_gen3_adastep.py)
- requirements.txt

## Real-robot data collection
```
python3 code/record_gen3_hdf5.py
```
Outputs HDF5 episodes in data/.

## Training
```
python3 train.py --task <task>
```
Checkpoints saved in checkpoints/<task>/ and dataset_stats.pkl generated.

## Deployment (ROS + ros_kortex)
```
roslaunch kortex_driver kortex_driver.launch robot_name:=my_gen3 ip_address:=<IP>
python3 code/deploy_gen3_adastep.py
```

Configure ROS params in deploy_gen3_adastep.py or via rosparam.
