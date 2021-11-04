#!/usr/bin/env python

from __future__ import print_function
from posixpath import join

from pybullet_tools.kuka_primitives import BodyPose, BodyConf, Command, get_grasp_gen, \
    get_ik_fn, get_free_motion_gen, get_holding_motion_gen
from pybullet_tools.utils import WorldSaver, enable_gravity, connect, dump_world, pairwise_collisions, set_joint_positions, set_pose, \
    draw_global_system, draw_pose, set_camera_pose, Pose, Point, set_default_camera, stable_z, \
    BLOCK_URDF, STOVE_URDF, load_model, wait_if_gui, disconnect, DRAKE_IIWA_URDF, \
    wait_if_gui, update_state, disable_real_time, HideOutput, set_joint_position



def plan(robot, block, fixed, teleport):
    grasp_gen = get_grasp_gen(robot, 'top')
    ik_fn = get_ik_fn(robot, fixed=fixed, teleport=teleport)
    free_motion_fn = get_free_motion_gen(robot, fixed=([block] + fixed), teleport=teleport)
    holding_motion_fn = get_holding_motion_gen(robot, fixed=fixed, teleport=teleport)

    pose0 = BodyPose(block)
    conf0 = BodyConf(robot)

    # import pdb
    # pdb.set_trace()
    saved_world = WorldSaver()
    for grasp, in grasp_gen(block):
        saved_world.restore()
        result1 = ik_fn(block, pose0, grasp)
        if result1 is None:
            continue
        conf1, path2 = result1
        pose0.assign()
        result2 = free_motion_fn(conf0, conf1)
        if result2 is None:
            continue
        path1, = result2
        result3 = holding_motion_fn(conf1, conf0, block, grasp)
        if result3 is None:
            continue
        path3, = result3
        return Command(path1.body_paths +
                          path2.body_paths +
                          path3.body_paths)
    return None


def main(display='execute'): # control | execute | step
    """
    NOTE: 
    by default, using models/drake/iiwa_description/urdf/iiwa14_polytope_collision.urdf
    This urdf misses collision objects for most links, manually added some collision objects for some links
    using primitive shapes, currently link 0 and link 5 is not including collision objects, otherwise not work

    NOTE: collision object of block consists of small balls at 4 
    corners and a slightly small cubic primitive, its self collision is by default True
    pairwise_collision(block, block) --> True

    Todo: 
    - perturb joint states and restore
    """
    
    connect(use_gui=True)
    disable_real_time()
    draw_global_system()
    with HideOutput():
        robot = load_model(DRAKE_IIWA_URDF) # KUKA_IIWA_URDF | DRAKE_IIWA_URDF
        floor = load_model('models/short_floor.urdf')
    block = load_model(BLOCK_URDF, fixed_base=False)
    set_pose(block, Pose(Point(y=0.5, z=stable_z(block, floor))))

    stove = load_model(STOVE_URDF, fixed_base=True)
    set_pose(stove, Pose(Point(y=0.1, z=stable_z(block, floor)+1)))

    set_default_camera(distance=2)
    # dump_world() # seems just print

    saved_world = WorldSaver()

    # collision checking
    from pybullet_tools.utils import pairwise_collisions, get_joints, single_collision, set_joint_positions, get_movable_joints, BodySaver
    import numpy as np
    def estimate_collision_risk(saved_body:BodySaver, positions=None, noise_scale=0.1, num_samples=500):
        joints = saved_body.conf_saver.joints
        robot = saved_body.body
        
        if positions is  None:
            positions = np.array(saved_body.conf_saver.positions)

        num_samples = float(num_samples)
        num_collision = 0.
        for i in range(int(num_samples)):
            noise = np.random.normal(loc=0, scale=noise_scale, size=len(positions))
            noisy_position = np.array(positions) + noise
            set_joint_positions(robot, joints=joints, values=noisy_position)
            if single_collision(robot):
                num_collision += 1.
        
        saved_body.restore()
        return num_collision / num_samples

    def forward_kinematics(saved_body:BodySaver, jointpositions=None, link=None):
        joints = saved_body.conf_saver.joints
        robot = saved_body.body

        if jointpositions is None:
            jointpositions = np.random.rand(len(joints))
        
        set_joint_positions(robot, joints=joints, values=jointpositions)
        if link is None:
            link = joints[-1]
        w_P_link, w_Q_link, _, _, _, _ = p.getLinkState(robot, link)

        saved_body.restore()
        return jointpositions, w_P_link, w_Q_link


    
    import pybullet as p
    p.setCollisionFilterPair(robot, robot, 1, 2, 0)
    p.setCollisionFilterPair(robot, robot, 2, 3, 0)
    p.setCollisionFilterPair(robot, robot, 3, 4, 0)
    p.setCollisionFilterPair(robot, robot, 4, 5, 0)
    p.setCollisionFilterPair(robot, robot, 5, 6, 0)
    p.setCollisionFilterPair(robot, robot, 6, 7, 0)


    w_P_link, w_Q_link, _, _, _, _ = p.getLinkState(robot, 8)




    from IPython import embed
    embed()

    js, tmp_p, tmp_q = forward_kinematics(saved_world.body_savers[0], link=8)
    js_p = p.calculateInverseKinematics(robot, 8, tmp_p, tmp_q)
    js_p, tmp_pp, tmp_qq = forward_kinematics(saved_world.body_savers[0], jointpositions=js_p, link=8)
    
    print('pos error: {}'.format( np.linalg.norm(np.array(tmp_p) - np.array(tmp_pp)) ) )
    print('ori error: {}'.format( np.linalg.norm(np.array(tmp_q) - np.array(tmp_qq)) ) )
    
    
    # get a set of joint states, for example, saved_world
    # joint_states = saved_world.body_savers[0].conf_saver.positions
    # joints = saved_world.body_savers[0].conf_saver.joints
    # # perturb
    # noise = np.random.normal(loc=0, scale=0.1, size=len(joint_states))
    # # joint_states = np.array(joint_states) + noise
    # set_joint_positions(robot, joints=joints, values=joint_states)

    # saved_world.restore()
    # saved_world.restore()
    estimate_collision_risk(saved_world.body_savers[0], 
                            # np.zeros(7,),
                            noise_scale=0.01,
                            num_samples=1000)


    # pairwise_collision(stove, block)
    command = plan(robot, block, fixed=[floor, stove], teleport=False)
    # import pdb
    # pdb.set_trace()
    if (command is None) or (display is None):
        print('Unable to find a plan!')
        return

    saved_world.restore()
    update_state()
    wait_if_gui('{}?'.format(display))
    if display == 'control':
        enable_gravity()
        command.control(real_time=False, dt=0)
    elif display == 'execute':
        command.refine(num_steps=10).execute(time_step=0.005)
    elif display == 'step':
        command.step()
    else:
        raise ValueError(display)

    print('Quit?')
    wait_if_gui()
    disconnect()

if __name__ == '__main__':
    main()