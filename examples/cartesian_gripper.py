from __future__ import print_function
from os import link
from posixpath import join

# from pybullet_tools.kuka_primitives import BodyPose, BodyConf, Command, get_grasp_gen, \
#     get_ik_fn, get_free_motion_gen, get_holding_motion_gen, plan_joint_motion
from pybullet_tools.utils import WorldSaver, enable_gravity, connect, dump_world, pairwise_collisions, plan_joint_motion, set_joint_positions, set_pose, \
    draw_global_system, draw_pose, set_camera_pose, Pose, Point, set_default_camera, stable_z, \
    BLOCK_URDF, STOVE_URDF, load_model, wait_if_gui, disconnect, \
    wait_if_gui, update_state, disable_real_time, HideOutput, set_joint_positions
    
from matplotlib import pylab

import pybullet as p
import numpy as np
import time
import gym
import pdb

class CartesianGripperEnv(object):
  """
  Implementation of CartesianGripper robot in PyBullet
  """
  def __init__(self, use_debug_camera=False, use_gui=True):
    SDF_path = 'wsg_50_description/sdf/schunk_wsg_50_welded_fingers.sdf'
    SDF_part_bottom = 'wsg_50_description/sdf/part_bottom.sdf'
    SDF_part_top = 'wsg_50_description/sdf/part_top.sdf'
    sim_id = connect(use_gui=True)
    disable_real_time()

    # DEBUG CAMERA SETTING
    if use_debug_camera:
      p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
      p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW,1)
      p.configureDebugVisualizer(p.COV_ENABLE_KEYBOARD_SHORTCUTS, 1)
      p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 1)
      p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 1)
      p.resetDebugVisualizerCamera(1.0, 0, -45, [0,0,0])  
    draw_global_system()
    
    self.robots = []

    with HideOutput():
        # robot = load_model(DRAKE_IIWA_URDF) # KUKA_IIWA_URDF | DRAKE_IIWA_URDF
        robot = p.loadSDF(SDF_path, 
                    # globalScaling=1., 
                    physicsClientId=0)

        p.resetBasePositionAndOrientation(robot[0], 
                                          [0,0,0.5], 
                                        [ -0.7068252, 0, 0, 0.7073883 ],
                                          )
        floor = load_model('models/short_floor.urdf', fixed_base=True)
        
        self.robots = [robot, floor]
    

    
    # block = load_model(BLOCK_URDF, 
    #                    pose=([0,0,0.1], [0,0,0,1]),
    #                    fixed_base=False)
    # set_pose(block, Pose(Point(y=0.5, z=stable_z(block, floor))))

    part_bottom = p.loadSDF(SDF_part_bottom, physicsClientId=0)
    self.robots.append(part_bottom)
    
    p.resetBasePositionAndOrientation(part_bottom[0], 
                                    [0,0,0.01], 
                                    [1, 0, 0, 0 ],
                                    )
    
    p.setGravity(0, 0, -10)
    
    self.action_space = gym.spaces.Box(low=np.array([-0.05, -0.05, -0.05]), 
                             high=np.array([0.05, 0.05, 0.05]), dtype=np.float)

    self.external_force = None
    
    ## NOTE: required for external force to work 
    p.setRealTimeSimulation(0)
    
  
  def setup_camera(self):
    """
    set up camera parameters
    """
    camera_params = {
      'camTargetPos': [0, 0, 0],
      'yaw': 0,
      'pitch': -45.0,
      'roll': 0,
      'upAxisIndex': 2,
      'camDistance': 1.5,
      'pixelWidth': 640,
      'pixelHeight': 480,
      'nearPlane': 0.01,
      'farPlane': 100,
      'fov': 60,
    }

    viewMatrix = p.computeViewMatrixFromYawPitchRoll(camera_params['camTargetPos'], 
                    camera_params['camDistance'], 
                    camera_params['yaw'], 
                    camera_params['pitch'],
                    camera_params['roll'], 
                    camera_params['upAxisIndex'])
    aspect = camera_params['pixelWidth'] / camera_params['pixelHeight']
    projectionMatrix = p.computeProjectionMatrixFOV(camera_params['fov'], 
                          aspect,
                          camera_params['nearPlane'], 
                          camera_params['farPlane'])
    
    self.camera_params = camera_params
    self.viewMatrix = viewMatrix
    self.aspect = aspect
    self.projectionMatrix = projectionMatrix
    
    return camera_params
  
  def get_image(self):
    img_arr = p.getCameraImage(self.camera_params['pixelWidth'],
        self.camera_params['pixelHeight'],
        self.viewMatrix,
        self.projectionMatrix,)
    return img_arr
  
  def save_image(self, saveloc=None):
    img_arr = self.get_image()
    w = img_arr[0]  #width of the image, in pixels
    h = img_arr[1]  #height of the image, in pixels
    rgb = img_arr[2]  #color data RGB
    dep = img_arr[3]  #depth data
    np_img_arr = np.reshape(rgb, (h, w, 4))
    np_img_arr = np_img_arr * (1. / 255.)
    # pylab.imshow(np_img_arr, interpolation='none', animated=True, label="pybullet")
    if saveloc:
      ## TODO: write RGBD file?
      np.save(saveloc, np_img_arr)

  def step(self, at:np.ndarray, stepsize=50, absolute=False, realtime=True):
    # get current joint state
    px, vx, fx, tx = p.getJointState(0, 0)
    py, vy, fy, ty = p.getJointState(0, 1)
    pz, vz, fz, tz = p.getJointState(0, 2)
    if not absolute:
      p.setJointMotorControl2(0, 0, p.POSITION_CONTROL, targetPosition=px + at[0])
      p.setJointMotorControl2(0, 1, p.POSITION_CONTROL, targetPosition=py + at[1])
      p.setJointMotorControl2(0, 2, p.POSITION_CONTROL, targetPosition=pz + at[2])
    else:
      p.setJointMotorControl2(0, 0, p.POSITION_CONTROL, targetPosition=at[0])
      p.setJointMotorControl2(0, 1, p.POSITION_CONTROL, targetPosition=at[1])
      p.setJointMotorControl2(0, 2, p.POSITION_CONTROL, targetPosition=at[2])
    for i in range (stepsize):
        p.stepSimulation()
        if realtime:
          time.sleep(1./240.)
    return
  
  def get_jointstate(self,):
    """
    get the joint state of the gripper
    """
    px, vx, fx, tx = p.getJointState(0, 0)
    py, vy, fy, ty = p.getJointState(0, 1)
    pz, vz, fz, tz = p.getJointState(0, 2)
    return np.array([px, py, pz])
  
  def apply_suction(self, 
                       body_id:int,
                       linkIndex=-1,
                       forceObj=[0,0,1],
                       posObj=[0,0,0],
                       flags=p.LINK_FRAME,
                       ):
    """
    apply external force from the center of the body to the center of the end effector surface
    """
    p.applyExternalForce(body_id, linkIndex, forceObj, posObj, flags)


if __name__ == '__main__':
  env = CartesianGripperEnv(use_debug_camera=True,
                        use_gui=True)
  env.setup_camera()
  for i in range(1000):
    # pdb.set_trace()
    at = env.action_space.sample()
    env.apply_suction(body_id=2, 
                      forceObj=[0,0,-15])
    # uncomment to get visual info
    env.get_image()
    env.step(at)
    
