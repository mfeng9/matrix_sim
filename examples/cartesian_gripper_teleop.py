## Keyboard controller used for generating
## human demonstrations
## adopted from 
## drake/examples/manipulation_station/
## end_effector_teleop_mouse.py

import pygame
import numpy as np
from pygame.locals import *
from copy import deepcopy
import sys
import os
import pickle
from cartesian_gripper import CartesianGripperEnv
import pdb

class TeleopMouseKeyboardManager():
    """exact copy and paste from 
    drake/examples/manipulation_station/
    end_effector_teleop_mouse.py

    This class does not depend on drake
    """

    def __init__(self, grab_focus=True):
        pygame.init()
        # We don't actually want a screen, but
        # I can't get this to work without a tiny screen.
        # Setting it to 1 pixel.
        screen_size = 1
        self.screen = pygame.display.set_mode((screen_size, screen_size))

        self.side_button_back_DOWN = False
        self.side_button_fwd_DOWN = False
        if grab_focus:
            self.grab_mouse_focus()

    def grab_mouse_focus(self):
        pygame.event.set_grab(True)
        pygame.mouse.set_visible(False)

    def release_mouse_focus(self):
        pygame.event.set_grab(False)
        pygame.mouse.set_visible(True)

    def get_events(self):
        mouse_wheel_up = mouse_wheel_down = False

        for event in pygame.event.get():
            if event.type == QUIT:
                sys.exit(0)
            elif event.type == KEYDOWN and event.key == K_ESCAPE:
                sys.exit(0)
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 4:
                    mouse_wheel_up = True
                if event.button == 5:
                    mouse_wheel_down = True
                if event.button == 8:
                    self.side_button_back_DOWN = True
                if event.button == 9:
                    self.side_button_fwd_DOWN = True
            if event.type == pygame.MOUSEBUTTONUP:
                if event.button == 8:
                    self.side_button_back_DOWN = False
                if event.button == 9:
                    self.side_button_fwd_DOWN = False

        keys = pygame.key.get_pressed()
        delta_x, delta_y = pygame.mouse.get_rel()
        left_mouse_button, _, right_mouse_button = pygame.mouse.get_pressed()

        if keys[K_RETURN]:
            self.grab_mouse_focus()
        if keys[K_SPACE]:
            self.release_mouse_focus()

        events = dict()
        events["delta_x"] = delta_x
        events["delta_y"] = delta_y
        events["w"] = keys[K_w]
        events["a"] = keys[K_a]
        events["s"] = keys[K_s]
        events["d"] = keys[K_d]
        events["r"] = keys[K_r]
        events["p"] = keys[K_p]
        events["q"] = keys[K_q]
        events["o"] = keys[K_o]
        events["e"] = keys[K_e]
        events["j"] = keys[K_j]
        events["l"] = keys[K_l]
        events["i"] = keys[K_i]
        events["k"] = keys[K_k]
        events["c"] = keys[K_c]
        events["z"] = keys[K_z]
        events["g"] = keys[K_g]
        events["h"] = keys[K_h]
        events["u"] = keys[K_u]
        events["r"] = keys[K_r]
        events["mouse_wheel_up"] = mouse_wheel_up
        events["mouse_wheel_down"] = mouse_wheel_down
        events["left_mouse_button"] = left_mouse_button
        events["right_mouse_button"] = right_mouse_button
        events["side_button_back"] = self.side_button_back_DOWN
        events["side_button_forward"] = self.side_button_fwd_DOWN
        return events

class CartesianGripperTeleController:
    def __init__(self, env:CartesianGripperEnv,
                 grab_focus=False,
                 data_dir = None,
                 dt = 0.05):
        """Simple keyboard controller for the PlanarGripper environment

        Args:
            env (drake planar environment): object of drake
            grab_focus (bool, optional): Pygame related, think it is to freeze
                                         and listen to mouse inputs. Defaults to False.
            dt (float, optional): simulation step in the keyboard event handler
                                  Need to see if this value can be different 
                                  from the dt value in the env step function.
                                  Defaults to 0.05.
        """
        self.env = env
        self.target_pos = env.get_jointstate()
        self.suction_state = False
        self.teleop_manager = TeleopMouseKeyboardManager(grab_focus=grab_focus)
        self.translation_step = 0.01
        self.rotation_step = 0.1

        self.dt = dt ## NOTE: teleop latency
        # data saving
        self.data_dir = data_dir
        self.controller_callback = self.HandleEvents
        

    def HandleEvents(self, events):
        print('rotation and translation step are the same, '
        'tune if rotation is too slow')
        if events['a']:
            self.target_pos[0] -= self.translation_step
        if events['d']:
            self.target_pos[0] += self.translation_step
        if events['w']:
            self.target_pos[2] -= self.translation_step
        if events['s']:
            self.target_pos[2] += self.translation_step
        if events['i']:
            self.target_pos[1] += self.translation_step
        if events['k']:
            self.target_pos[1] -= self.translation_step
        if events['g']:
            self.suction_state = True
        if events['h']:
            self.suction_state = False
        if events['o']:
            self.save_history(self.data_dir)
        if events['p']:
            self.env.reset()
    
    def start_teleop(self):
        """allows potential smoother tele-op by directly setting 
        the desired state in the controller, by passing the
        gym-like stepping mechanism, not good for generating 
        RL training demonstration, as the actions are not the real
        actions
        """
        self.history = []
        while True:
            events = self.teleop_manager.get_events()
            self.controller_callback(events)
            if self.suction_state:
                self.env.apply_suction(body_id=2, 
                                forceObj=[0,0,-15])
            self.env.step(at=self.target_pos, absolute=True)
    
    def save_history(self, data_dir, filename=None):
        if filename is None:
            filename = input('enter the name of the file (without file extension .p)\n')
            filename = filename + '.p'
        target_full_path = os.path.join(data_dir, filename)
        pickle.dump(self.history, open(target_full_path, 'wb'))
        print('teleop data saved at: {}'.format(filename))
        if isinstance(self.history, dict):
            print('number of frames: {}'.format(len(self.history['reward'])))
        else:
            print('number of frames: {}'.format(len(self.history)))
    
if __name__ == '__main__':
    
    env = CartesianGripperEnv(use_debug_camera=True, use_gui=True)
    teleop = CartesianGripperTeleController(env,
            # data_path="/home/night/apps/rl/mbrl-gpmm/experiments/data/"
            # "teleop_simple_moves.p",
            dt=0.1)
    teleop.start_teleop()

