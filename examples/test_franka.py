#!/usr/bin/env python

from __future__ import print_function

import pybullet as p

from pybullet_tools.utils import add_data_path, connect, dump_body, disconnect, wait_for_user, \
    get_movable_joints, get_sample_fn, set_joint_positions, get_joint_name, LockRenderer, link_from_name, get_link_pose, \
    multiply, Pose, Point, interpolate_poses, HideOutput, draw_pose, set_camera_pose, load_pybullet, \
    assign_link_colors

from pybullet_tools.ikfast.franka_panda.ik import PANDA_INFO, FRANKA_URDF
from pybullet_tools.ikfast.ikfast import get_ik_joints, either_inverse_kinematics, print_ik_warning


def test_retraction(robot, info, tool_link, distance=0.1, **kwargs):
    ik_joints = get_ik_joints(robot, info, tool_link)
    start_pose = get_link_pose(robot, tool_link)
    end_pose = multiply(start_pose, Pose(Point(z=-distance)))
    for pose in interpolate_poses(start_pose, end_pose, pos_step_size=0.01):
        conf = next(either_inverse_kinematics(robot, info, tool_link, pose, **kwargs), None)
        if conf is None:
            print('Failure!')
            wait_for_user()
            break
        set_joint_positions(robot, ik_joints, conf)
        wait_for_user()
        # for conf in islice(ikfast_inverse_kinematics(robot, info, tool_link, pose, max_attempts=INF, max_distance=0.5), 1):
        #    set_joint_positions(robot, joints[:len(conf)], conf)
        #    wait_for_user()


#####################################

def main():
    connect(use_gui=True)
    add_data_path()
    draw_pose(Pose(), length=1.)
    set_camera_pose(camera_point=[1, -1, 1])

    plane = p.loadURDF("plane.urdf")
    with LockRenderer():
        with HideOutput():
            robot = load_pybullet(FRANKA_URDF, fixed_base=True)
            assign_link_colors(robot)
            #set_all_color(robot, GREEN)

    dump_body(robot)
    print('Start?')
    wait_for_user()

    info = PANDA_INFO
    tool_link = link_from_name(robot, 'panda_hand')
    draw_pose(Pose(), parent=robot, parent_link=tool_link)
    joints = get_movable_joints(robot)
    print('Joints', [get_joint_name(robot, joint) for joint in joints])
    print_ik_warning(info)

    sample_fn = get_sample_fn(robot, joints)
    for i in range(10):
        print('Iteration:', i)
        conf = sample_fn()
        set_joint_positions(robot, joints, conf)
        wait_for_user()
        test_retraction(robot, info, tool_link, max_distance=0.01, max_time=0.05, max_candidates=100)
    disconnect()

if __name__ == '__main__':
    main()
