import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
                       For this custom task, since starting location is the ground at the center of 
                       the x-y plane, we set init_pose to np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).
                       
                       
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
                       This custom task begins with the copter at rest, so we set 
                       init_velocities to np.array([0.0, 0.0, 0.0]).
                             
            init_angle_velocities: initial radians/second for each of the three Euler angles

                       This custom task begins with the copter at rest, so we also set 
                       init_angle_velocities to np.array([0.0, 0.0, 0.0]).
                       
            runtime: time limit for each episode. Set this large enough to give the agent time to 
                     reach the target position.
                     
            target_pos: target/goal (x,y,z) position for the agent. This task's goal is for the 
                        quadcopter to reach a height of N meters above the x-y plane's center, 
                        so target_pos is set to np.array([0., 0., N.]).
            
            
            This task will be a straightforward takeoff task: the quadcopter begins 
            the task at rest on the ground at the center of the map -- at x-y-z coordinates 
            of (0,0,0). Because the copter begins at rest, its initial velocities 
            and its initial angular velocities are both 0.
 
            Once the task commences, the quadcopter must takeoff and elevate to an altitude 
            of N meters above the ground as rapidly as possible. It should do this while 
            maintaining its location above the center of the x-y plane. 

            The quadcopter's target position will thus be (0,0,100).
        
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3
    
        # State size for this task contains only x-y-z coordinates.
        self.state_size = self.action_repeat * 6
        
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal
        if target_pos is None :
            print("Setting default init pose")
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.])

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        
        # Penalize the copter when it is far away from its target position. 
        # The closer the copter gets to its target, the smaller this penalty becomes.
        
        # original provided reward:
        # reward = 1.-.3*(abs(self.sim.pose[:3] - self.target_pos)).sum() 
        # tanh function used:
        reward = np.tanh(1 - 0.002*(abs(self.sim.pose[:3] - self.target_pos))).sum()
       
        # reward = 1 - 0.3*(abs(self.sim.pose[:3] - self.target_pos)).sum()
        # reward = 1.-.3*(abs(self.sim.pose[:3] - self.target_pos)).sum()
        # reward = 1 - .1*((abs(self.sim.pose[:3] - self.target_pos)).sum())**0.4
        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
            # State size for this task contains only x-y-z coordinates.
            pose_all.append(self.sim.pose)
            
            # if self.sim.pose[:3] >= self.target_pos:  # agent has crossed the target height
            #    reward += 1.0  # bonus reward
            # approach to target
            if self.sim.pose[2] >= 10:
                reward += 5    
            
            # good approach to target
            if self.sim.pose[2] >= (self.target_pos[2]-50):
                reward += 10
                
            # when within +/- 5 of target pose  (close to target)
            if (self.sim.pose[2] >= self.target_pos[2]-5) and (self.sim.pose[2] <= self.target_pos[2] + 5):
                   reward += 5
            if done :
                reward += 10
 
 
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        
        self.sim.reset()
        # State size for this task contains only x-y-z coordinates.
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state