#!/usr/bin/env python
import torch
from numpy import cos, sin, tan, clip
from Agent.world_model.agent_model.KinematicBicycleModel.libs.normalise_angle import normalise_angle

class Car:

    def __init__(self, init_x, init_y, init_yaw, dt):

        # Model parameters
        self.x = init_x
        self.y = init_y
        self.yaw = init_yaw
        self.v = 0.0
        self.delta = 0.0
        self.omega = 0.0
        self.wheelbase = 2.96
        self.max_steer = np.deg2rad(60)
        self.dt = dt
        self.c_r = 0.01
        self.c_a = 2.0

        # Description parameters
        self.overall_length = 4.97
        self.overall_width = 1.964
        self.tyre_diameter = 0.4826
        self.tyre_width = 0.2032
        self.axle_track = 1.662
        self.rear_overhang = (self.overall_length - self.wheelbase) / 2
        self.colour = 'black'

        self.kbm = KinematicBicycleModel(self.wheelbase, self.max_steer, self.dt, self.c_r, self.c_a)

    def drive(self, throttle, steer):
        
        throttle = rand.uniform(150, 200)
        self.delta = steer
        self.x, self.y, self.yaw, self.v, _, _ = self.kbm.kinematic_model(self.x, self.y, self.yaw, self.v, throttle, self.delta)


class KinematicBicycleModel():

    def __init__(self, wheelbase=1.0, max_steer=0.7, dt=0.05, c_r=0.0, c_a=0.0):
        """
        2D Kinematic Bicycle Model

        At initialisation
        :param wheelbase:       (float) vehicle's wheelbase [m]
        :param max_steer:       (float) vehicle's steering limits [rad]
        :param dt:              (float) discrete time period [s]
        :param c_r:             (float) vehicle's coefficient of resistance 
        :param c_a:             (float) vehicle's aerodynamic coefficient
    
        At every time step  
        :param x:               (float) vehicle's x-coordinate [m]
        :param y:               (float) vehicle's y-coordinate [m]
        :param yaw:             (float) vehicle's heading [rad]
        :param velocity:        (float) vehicle's velocity in the x-axis [m/s]
        :param throttle:        (float) vehicle's accleration [m/s^2]
        :param delta:           (float) vehicle's steering angle [rad]
    
        :return x:              (float) vehicle's x-coordinate [m]
        :return y:              (float) vehicle's y-coordinate [m]
        :return yaw:            (float) vehicle's heading [rad]
        :return velocity:       (float) vehicle's velocity in the x-axis [m/s]
        :return delta:          (float) vehicle's steering angle [rad]
        :return omega:          (float) vehicle's angular velocity [rad/s]
        """

        self.dt = dt
        self.wheelbase = wheelbase
        self.max_steer = max_steer
        self.c_r = c_r
        self.c_a = c_a

    def kinematic_model(self, x, y, yaw, velocity, throttle, delta):
        # Compute the local velocity in the x-axis
        f_load = velocity * (self.c_r + self.c_a * velocity)

        velocity += self.dt * (throttle - f_load)

        # Compute the radius and angular velocity of the kinematic bicycle model
        delta = clip(delta, -self.max_steer, self.max_steer)

        # Compute the state change rate
        x_dot = velocity * cos(yaw)
        y_dot = velocity * sin(yaw)
        omega = velocity * tan(delta) / self.wheelbase

        # Compute the final state using the discrete time model
        x += x_dot * self.dt
        y += y_dot * self.dt
        yaw += omega * self.dt
        yaw = normalise_angle(yaw)
        
        return x, y, yaw, velocity, delta, omega
    
class KinematicBicycleModel_Pytorch():

    def __init__(self, wheelbase=1.0, max_steer=0.7, dt=0.05, c_r=0.0, c_a=0.0):
        """
        2D Kinematic Bicycle Model

        At initialisation
        :param wheelbase:       (float) vehicle's wheelbase [m]
        :param max_steer:       (float) vehicle's steering limits [rad]
        :param dt:              (float) discrete time period [s]
        :param c_r:             (float) vehicle's coefficient of resistance 
        :param c_a:             (float) vehicle's aerodynamic coefficient
    
        At every time step  
        :param x:               (float) vehicle's x-coordinate [m]
        :param y:               (float) vehicle's y-coordinate [m]
        :param yaw:             (float) vehicle's heading [rad]
        :param velocity:        (float) vehicle's velocity in the x-axis [m/s]
        :param throttle:        (float) vehicle's accleration [m/s^2]
        :param delta:           (float) vehicle's steering angle [rad]
    
        :return x:              (float) vehicle's x-coordinate [m]
        :return y:              (float) vehicle's y-coordinate [m]
        :return yaw:            (float) vehicle's heading [rad]
        :return velocity:       (float) vehicle's velocity in the x-axis [m/s]
        :return delta:          (float) vehicle's steering angle [rad]
        :return omega:          (float) vehicle's angular velocity [rad/s]
        """

        self.dt = dt
        self.wheelbase = wheelbase
        self.max_steer = max_steer
        self.c_r = c_r
        self.c_a = c_a

    def kinematic_model(self, x, y, yaw, velocity, throttle, delta):
        # Compute the local velocity in the x-axis
        ca = torch.mul(velocity, self.c_a)
        temp = torch.add(ca, self.c_r)
        f_load = torch.mul(velocity, temp) # 
        
        dv = torch.mul(torch.sub(throttle, f_load), self.dt)
        velocity = torch.add(velocity, dv)  

        # Compute the state change rate
        x_dot = torch.mul(velocity, torch.cos(yaw))
        y_dot = torch.mul(velocity, torch.sin(yaw))
        omega = torch.mul(velocity, torch.tan(delta))
        omega = torch.mul(omega, 1/self.wheelbase)
        
        # Compute the final state using the discrete time model
        x = torch.add(x, torch.mul(x_dot, self.dt))
        y = torch.add(y, torch.mul(y_dot, self.dt))
        yaw = torch.add(yaw, torch.mul(omega, self.dt))
        yaw = torch.atan2(torch.sin(yaw), torch.cos(yaw))

        return x, y, yaw, velocity, delta, omega

def main():

    print("This script is not meant to be executable, and should be used as a library.")

if __name__ == "__main__":
    main()
