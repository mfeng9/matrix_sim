from __future__ import print_function
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


SDF_path = '/home/night/apps/baselines/pybullet-planning/examples/schunk_wsg_50_welded_fingers.sdf'
SDF_part_bottom = '/home/night/apps/baselines/pybullet-planning/examples/wsg_50_description/sdf/part_bottom.sdf'
SDF_part_top = '/home/night/apps/baselines/pybullet-planning/examples/wsg_50_description/sdf/part_top.sdf'
# DRAKE_IIWA_URDF = 'models/drake/iiwa_description/urdf/iiwa14_primitive_collision.urdf'


sim_id = connect(use_gui=True)
disable_real_time()


# DEBUG CAMERA SETTING
# p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
# p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW,1)
# p.configureDebugVisualizer(p.COV_ENABLE_KEYBOARD_SHORTCUTS, 1)
# p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 1)
# p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 1)
# p.resetDebugVisualizerCamera(1.0, 0, -45, [0,0,0])  
draw_global_system()

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
    

    
# block = load_model(BLOCK_URDF, 
#                    pose=([0,0,0.1], [0,0,0,1]),
#                    fixed_base=False)
# set_pose(block, Pose(Point(y=0.5, z=stable_z(block, floor))))

part_bottom = p.loadSDF(SDF_part_bottom, physicsClientId=0)
p.resetBasePositionAndOrientation(part_bottom[0], 
                                [0,0,0.01], 
                                [1, 0, 0, 0 ],
                                )


camTargetPos = [0, 0, 0]
cameraUp = [0, 0, 1]
cameraPos = [1, 0, 0]
p.setGravity(0, 0, -10)
yaw = 0

pitch = -45.0
roll = 0
upAxisIndex = 2
camDistance = 1.5
pixelWidth = 640
pixelHeight = 480
nearPlane = 0.01
farPlane = 100
fov = 60
viewMatrix = p.computeViewMatrixFromYawPitchRoll(camTargetPos, camDistance, yaw, pitch,
                                                            roll, upAxisIndex)
aspect = pixelWidth / pixelHeight
projectionMatrix = p.computeProjectionMatrixFOV(fov, aspect, nearPlane, farPlane)
    
img_arr = p.getCameraImage(pixelWidth,pixelHeight,viewMatrix,projectionMatrix)
# TODO: save image
# w = img_arr[0]  #width of the image, in pixels
# h = img_arr[1]  #height of the image, in pixels
# rgb = img_arr[2]  #color data RGB
# dep = img_arr[3]  #depth data
# print("w=",w,"h=",h)
# np_img_arr = np.reshape(rgb, (h, w, 4))
# np_img_arr = np_img_arr * (1. / 255.)
# pylab.imshow(np_img_arr, interpolation='none', animated=True, label="pybullet")


def step(at:np.ndarray, stepsize=50, realtime=True):
  # TODO: add suck action
  # get current joint state
  px, vx, fx, tx = p.getJointState(0, 0)
  py, vy, fy, ty = p.getJointState(0, 1)
  pz, vz, fz, tz = p.getJointState(0, 2)
  p.setJointMotorControl2(0, 0, p.POSITION_CONTROL, targetPosition=px + at[0])
  p.setJointMotorControl2(0, 1, p.POSITION_CONTROL, targetPosition=py + at[1])
  p.setJointMotorControl2(0, 2, p.POSITION_CONTROL, targetPosition=pz + at[2])
  for i in range (stepsize):
      p.stepSimulation()
      if realtime:
        time.sleep(1./240.)
  return

action_space = gym.spaces.Box(low=np.array([-0.05, -0.05, -0.05]), 
                             high=np.array([0.05, 0.05, 0.05]), dtype=np.float)

for i in range(1000):
  at = action_space.sample()
  # uncomment to get visual info
  img_arr = p.getCameraImage(pixelWidth,pixelHeight,viewMatrix,projectionMatrix)
  step(at)
  
