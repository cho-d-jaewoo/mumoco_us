import h5py
import numpy as np

# 1. Open the HDF5 file in read-only mode ('r')
with h5py.File('demos/demo_0.hdf5', 'r') as f:
    
    # 2. Print all top-level keys (group or dataset names)
    #print("Keys in the file:", list(f.keys()))
    # 'actions' or 'obs'
    
    # 3. Access a specific dataset
    actions = f['actions']
    obs = f['obs']

    #print(list(obs.keys()))
    actions_ee_vel = actions['ee_vel']
    actions_gripper_action = actions['gripper_action']
    actions_joint_vel = actions['joint_vel']
    obs_ee_state = obs['ee_state']
    obs_gripper_img_rgb = obs['gripper_img_rgb']
    obs_gripper_state = obs['gripper_state']
    obs_joint_state = obs['joint_state']
    obs_left_realsense_img_depth = obs['left_realsense_img_depth']
    obs_left_realsense_img_rgb = obs['left_realsense_img_rgb']
    obs_right_realsense_img_depth = obs['right_realsense_img_depth']
    obs_right_realsense_img_rgb = obs['right_realsense_img_rgb']
    obs_timestep = obs['timestep']
    
    # 4. Check metadata without loading the entire array into memory
    print("Dataset shape:", actions_ee_vel.shape)
    print("Dataset type:", actions_ee_vel.dtype)
    
    # 5. Read the data into a NumPy array
    # Slicing [()] reads the entire dataset
    print(obs_left_realsense_img_depth[()])
    
    # Alternatively, slice a portion to save memory
    # partial_array = dataset[:100, :100]

print("Data loaded successfully!")
