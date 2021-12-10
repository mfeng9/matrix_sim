#!/usr/bin/env python

from __future__ import print_function
from posixpath import join

from pybullet_tools.kuka_primitives import BodyPose, BodyConf, Command, get_grasp_gen, \
    get_ik_fn, get_free_motion_gen, get_holding_motion_gen
from pybullet_tools.utils import WorldSaver, enable_gravity, connect, dump_world, pairwise_collisions, set_joint_positions, set_pose, \
    draw_global_system, draw_pose, set_camera_pose, Pose, Point, set_default_camera, stable_z, \
    BLOCK_URDF, STOVE_URDF, load_model, wait_if_gui, disconnect, \
    wait_if_gui, update_state, disable_real_time, HideOutput, set_joint_positions

import pybullet as p
import numpy as np


DRAKE_IIWA_URDF = 'models/drake/iiwa_description/urdf/iiwa14_polytope_collision.urdf'
# DRAKE_IIWA_URDF = 'models/drake/iiwa_description/urdf/iiwa14_primitive_collision.urdf'


connect(use_gui=True)
disable_real_time()
draw_global_system()
with HideOutput():
    # robot = load_model(DRAKE_IIWA_URDF) # KUKA_IIWA_URDF | DRAKE_IIWA_URDF
    robot = p.loadURDF(DRAKE_IIWA_URDF, useFixedBase=True, flags=p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS,
                              globalScaling=1., physicsClientId=0)

    floor = load_model('models/short_floor.urdf')
block = load_model(BLOCK_URDF, fixed_base=False)
set_pose(block, Pose(Point(y=0.5, z=stable_z(block, floor))))

set_default_camera(distance=2)
# dump_world() # seems just print
saved_world = WorldSaver()

robot_body = saved_world.body_savers[0]
joints = robot_body.conf_saver.joints
target_pos = np.array( [0.0, 0.6, 0.3] )
# joint_pos = p.calculateInverseKinematics(robot, 8, target_pos)
# set_joint_positions(robot, joints, joint_pos)
# p.stepSimulation()

def show_ik(target_pos):
    # target_pos = np.array( [0.0, 0.6, 0.3] )
    joint_pos = p.calculateInverseKinematics(robot, 8, target_pos)
    set_joint_positions(robot, joints, joint_pos)
    for i in range(3):
        p.stepSimulation()
ik_fn = get_ik_fn(robot, fixed=[block, floor], teleport=False)
free_motion_fn = get_free_motion_gen(robot, fixed=([block]), teleport=False)

grasp_gen = get_grasp_gen(robot, 'top')
pose0 = BodyPose(block)
conf0 = BodyConf(robot)
grasp = grasp_gen(block).__next__()[0]

while True:
    try:
        result1 = ik_fn(block, pose0, grasp)
        break
    except:
        pass
conf1, path2 = result1
pose0.assign()
result2 = free_motion_fn(conf0, conf1)

from IPython import embed
embed()


print('end')