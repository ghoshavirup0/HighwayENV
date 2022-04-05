from __future__ import division, print_function

from typing import Tuple

import copy

import numpy as np
import pandas as pd
from collections import deque
import matplotlib.pyplot as plt

from highway_env.road.road import Road
from highway_env.utils import Vector
from highway_env.vehicle.kinematics import Vehicle



from NGSIM_env import utils
from NGSIM_env.logger import Loggable





class Vehicle(Loggable):
    """
    A moving vehicle on a road, and its dynamics.

    The vehicle is represented by a dynamical system: a modified bicycle model.
    It's state is propagated depending on its steering and acceleration actions.
    """
    # Enable collision detection between vehicles 
    COLLISIONS_ENABLED = True
    # Vehicle length [m]
    LENGTH = 5.0
    # Vehicle width [m]
    WIDTH = 2.0
    # Range for random initial velocities [m/s]
    DEFAULT_VELOCITIES = [23, 25]
    # Maximum reachable velocity [m/s]
    MAX_VELOCITY = 40

    def __init__(self, road, position, heading=0, velocity=0):
        self.road = road
        self.position = np.array(position).astype('float')
        self.heading = heading
        self.velocity = velocity
        self.lane_index = self.road.network.get_closest_lane_index(self.position) if self.road else np.nan
        self.lane = self.road.network.get_lane(self.lane_index) if self.road else None
        self.action = {'steering': 0, 'acceleration': 0}
        self.crashed = False
        self.log = []
        self.history = deque(maxlen=50)

    @classmethod
    def make_on_lane(cls, road, lane_index, longitudinal, velocity=0):
        """
        Create a vehicle on a given lane at a longitudinal position.

        :param road: the road where the vehicle is driving
        :param lane_index: index of the lane where the vehicle is located
        :param longitudinal: longitudinal position along the lane
        :param velocity: initial velocity in [m/s]
        :return: A vehicle with at the specified position
        """
        lane = road.network.get_lane(lane_index)

        if velocity is None:
            velocity = lane.speed_limit

        return cls(road, lane.position(longitudinal, 0), lane.heading_at(longitudinal), velocity)

    @classmethod
    def create_random(cls, road, velocity=None, spacing=1):
        """
        Create a random vehicle on the road.

        The lane and /or velocity are chosen randomly, while longitudinal position is chosen behind the last
        vehicle in the road with density based on the number of lanes.

        :param road: the road where the vehicle is driving
        :param velocity: initial velocity in [m/s]. If None, will be chosen randomly
        :param spacing: ratio of spacing to the front vehicle, 1 being the default
        :return: A vehicle with random position and/or velocity
        """
        if velocity is None:
            velocity = road.np_random.uniform(Vehicle.DEFAULT_VELOCITIES[0], Vehicle.DEFAULT_VELOCITIES[1])

        default_spacing = 1.5*velocity

        _from = road.np_random.choice(list(road.network.graph.keys()))
        _to = road.np_random.choice(list(road.network.graph[_from].keys()))
        _id = road.np_random.choice(len(road.network.graph[_from][_to]))

        offset = spacing * default_spacing * np.exp(-5 / 30 * len(road.network.graph[_from][_to]))
        x0 = np.max([v.position[0] for v in road.vehicles]) if len(road.vehicles) else 3*offset
        x0 += offset * road.np_random.uniform(0.9, 1.1)
        
        v = cls(road,
                road.network.get_lane((_from, _to, _id)).position(x0, 0),
                road.network.get_lane((_from, _to, _id)).heading_at(x0),
                velocity)

        return v

    @classmethod
    def create_from(cls, vehicle):
        """
        Create a new vehicle from an existing one.
        Only the vehicle dynamics are copied, other properties are default.

        :param vehicle: a vehicle
        :return: a new vehicle at the same dynamical state
        """
        v = cls(vehicle.road, vehicle.position, vehicle.heading, vehicle.velocity)

        return v

    def act(self, action=None):
        """
        Store an action to be repeated.

        :param action: the input action
        """
        if action:
            self.action = action

    def step(self, dt):
        """
        Propagate the vehicle state given its actions.

        Integrate a modified bicycle model with a 1st-order response on the steering wheel dynamics.
        If the vehicle is crashed, the actions are overridden with erratic steering and braking until complete stop.
        The vehicle's current lane is updated.

        :param dt: timestep of integration of the model [s]
        """
        if self.crashed:
            self.action['steering'] = 0
            self.action['acceleration'] = -1.0*self.velocity
        self.action['steering'] = float(self.action['steering'])
        self.action['acceleration'] = float(self.action['acceleration'])
        
        if self.velocity > self.MAX_VELOCITY:
            self.action['acceleration'] = min(self.action['acceleration'], 1.0*(self.MAX_VELOCITY - self.velocity))
        elif self.velocity < -self.MAX_VELOCITY:
            self.action['acceleration'] = max(self.action['acceleration'], 1.0*(self.MAX_VELOCITY - self.velocity))

        v = self.velocity * np.array([np.cos(self.heading), np.sin(self.heading)])
        self.position += v * dt
        self.heading += self.velocity * np.tan(self.action['steering']) / self.LENGTH * dt
        self.velocity += self.action['acceleration'] * dt

        if self.road:
            self.lane_index = self.road.network.get_closest_lane_index(self.position)
            self.lane = self.road.network.get_lane(self.lane_index)
            if self.road.record_history:
                self.history.appendleft(self.create_from(self))

    def lane_distance_to(self, vehicle):
        """
        Compute the signed distance to another vehicle along current lane.

        :param vehicle: the other vehicle
        :return: the distance to the other vehicle [m]
        """
        if not vehicle:
            return np.nan
        
        return self.lane.local_coordinates(vehicle.position)[0] - self.lane.local_coordinates(self.position)[0]

    def check_collision(self, other):
        """
        Check for collision with another vehicle.

        :param other: the other vehicle
        """
        if not self.COLLISIONS_ENABLED or not other.COLLISIONS_ENABLED or self.crashed or other is self:
            return

        # Fast spherical pre-check
        if np.linalg.norm(other.position - self.position) > self.LENGTH:
            return

        # Accurate rectangular check
        if utils.rotated_rectangles_intersect((self.position, 0.9*self.LENGTH, 0.9*self.WIDTH, self.heading),
                                              (other.position, 0.9*other.LENGTH, 0.9*other.WIDTH, other.heading)):
            self.velocity = other.velocity = min([self.velocity, other.velocity], key=abs)
            self.crashed = other.crashed = True

    @property
    def direction(self):
        return np.array([np.cos(self.heading), np.sin(self.heading)])

    @property
    def destination(self):
        if getattr(self, "route", None):
            last_lane = self.road.network.get_lane(self.route[-1])
            return last_lane.position(last_lane.length, 0)
        else:
            return self.position

    @property
    def destination_direction(self):
        if (self.destination != self.position).any():
            return (self.destination - self.position) / np.linalg.norm(self.destination - self.position)
        else:
            return np.zeros((2,))

    @property
    def on_road(self):
        """ Is the vehicle on its current lane, or off-road ? """
        return self.lane.on_lane(self.position)

    def front_distance_to(self, other):
        return self.direction.dot(other.position - self.position)

    def to_dict(self, origin_vehicle=None, observe_intentions=True):
        #print(self.position)
        d = {
            'presence': 1,
            'x': self.position[0],
            'y': self.position[1],
            'vx': self.velocity * self.direction[0],
            'vy': self.velocity * self.direction[1],
            'cos_h': self.direction[0],
            'sin_h': self.direction[1],
            'cos_d': self.destination_direction[0],
            'sin_d': self.destination_direction[1]
        }
        if not observe_intentions:
            d["cos_d"] = d["sin_d"] = 0
        if origin_vehicle:
            origin_dict = origin_vehicle.to_dict()
            for key in ['x', 'y', 'vx', 'vy']:
                d[key] -= origin_dict[key]

        return d

    def dump(self):
        """
        Update the internal log of the vehicle, containing:
        - its kinematics;
        - some metrics relative to its neighbour vehicles.
        """
        data = {
            'x': self.position[0],
            'y': self.position[1],
            'psi': self.heading,
            'vx': self.velocity * np.cos(self.heading),
            'vy': self.velocity * np.sin(self.heading),
            'v': self.velocity,
            'acceleration': self.action['acceleration'],
            'steering': self.action['steering']}

        if self.road:
            for lane_index in self.road.network.side_lanes(self.lane_index):
                lane_coords = self.road.network.get_lane(lane_index).local_coordinates(self.position)
                data.update({
                    'dy_lane_{}'.format(lane_index): lane_coords[1],
                    'psi_lane_{}'.format(lane_index): self.road.network.get_lane(lane_index).heading_at(lane_coords[0])
                })
            front_vehicle, rear_vehicle = self.road.neighbour_vehicles(self)
            if front_vehicle:
                data.update({
                    'front_v': front_vehicle.velocity,
                    'front_distance': self.lane_distance_to(front_vehicle)
                })
            if rear_vehicle:
                data.update({
                    'rear_v': rear_vehicle.velocity,
                    'rear_distance': rear_vehicle.lane_distance_to(self)
                })

        self.log.append(data)

    def get_log(self):
        """
        Cast the internal log as a DataFrame.

        :return: the DataFrame of the Vehicle's log.
        """
        return pd.DataFrame(self.log)

    def __str__(self):
        return "{} #{}: {}".format(self.__class__.__name__, id(self) % 1000, self.position)

    def __repr__(self):
        return self.__str__()


class Obstacle(Vehicle):
    """
    A motionless obstacle at a given position.
    """

    def __init__(self, road, position, heading=0):
        super(Obstacle, self).__init__(road, position, velocity=0, heading=heading)
        self.target_velocity = 0
        self.LENGTH = self.WIDTH



class BicycleVehicle(Vehicle):
    """
    A dynamical bicycle model, with tire friction and slipping.
    
    See Chapter 2 of Lateral Vehicle Dynamics. Vehicle Dynamics and Control. Rajamani, R. (2011)
    """
    MASS: float = 1  # [kg]
    LENGTH_A: float = Vehicle.LENGTH / 2  # [m]
    LENGTH_B: float = Vehicle.LENGTH / 2  # [m]
    INERTIA_Z: float = 1/12 * MASS * (Vehicle.LENGTH ** 2 + Vehicle.WIDTH ** 2)  # [kg.m2]
    FRICTION_FRONT: float = 15.0 * MASS  # [N]
    FRICTION_REAR: float = 15.0 * MASS  # [N]

    MAX_ANGULAR_SPEED: float = 2 * np.pi  # [rad/s]
    MAX_SPEED: float = 15  # [m/s]

    def __init__(self, road: Road, position: Vector, heading: float = 0, speed: float = 0) -> None:
        super().__init__(road, position, heading, speed)
        self.lateral_speed = 0
        self.yaw_rate = 0
        self.theta = None
        self.A_lat, self.B_lat = self.lateral_lpv_dynamics()

    @property
    def state(self) -> np.ndarray:
        return np.array([[self.position[0]],
                         [self.position[1]],
                         [self.heading],
                         [self.speed],
                         [self.lateral_speed],
                         [self.yaw_rate]])

    @property
    def derivative(self) -> np.ndarray:
        """
        See Chapter 2 of Lateral Vehicle Dynamics. Vehicle Dynamics and Control. Rajamani, R. (2011)

        :return: the state derivative
        """
        delta_f = self.action["steering"]
        delta_r = 0
        theta_vf = np.arctan2(self.lateral_speed + self.LENGTH_A * self.yaw_rate, self.speed)  # (2.27)
        theta_vr = np.arctan2(self.lateral_speed - self.LENGTH_B * self.yaw_rate, self.speed)  # (2.28)
        f_yf = 2*self.FRICTION_FRONT * (delta_f - theta_vf)  # (2.25)
        f_yr = 2*self.FRICTION_REAR * (delta_r - theta_vr)  # (2.26)
        if abs(self.speed) < 1:  # Low speed dynamics: damping of lateral speed and yaw rate
            f_yf = - self.MASS * self.lateral_speed - self.INERTIA_Z/self.LENGTH_A * self.yaw_rate
            f_yr = - self.MASS * self.lateral_speed + self.INERTIA_Z/self.LENGTH_A * self.yaw_rate
        d_lateral_speed = 1/self.MASS * (f_yf + f_yr) - self.yaw_rate * self.speed  # (2.21)
        d_yaw_rate = 1/self.INERTIA_Z * (self.LENGTH_A * f_yf - self.LENGTH_B * f_yr)  # (2.22)
        c, s = np.cos(self.heading), np.sin(self.heading)
        R = np.array(((c, -s), (s, c)))
        speed = R @ np.array([self.speed, self.lateral_speed])
        return np.array([[speed[0]],
                         [speed[1]],
                         [self.yaw_rate],
                         [self.action['acceleration']],
                         [d_lateral_speed],
                         [d_yaw_rate]])

    @property
    def derivative_linear(self) -> np.ndarray:
        """
        Linearized lateral dynamics.
            
        This model is based on the following assumptions:
        - the vehicle is moving with a constant longitudinal speed
        - the steering input to front tires and the corresponding slip angles are small
        
        See https://pdfs.semanticscholar.org/bb9c/d2892e9327ec1ee647c30c320f2089b290c1.pdf, Chapter 3.
        """
        x = np.array([[self.lateral_speed], [self.yaw_rate]])
        u = np.array([[self.action['steering']]])
        self.A_lat, self.B_lat = self.lateral_lpv_dynamics()
        dx = self.A_lat @ x + self.B_lat @ u
        c, s = np.cos(self.heading), np.sin(self.heading)
        R = np.array(((c, -s), (s, c)))
        speed = R @ np.array([self.speed, self.lateral_speed])
        return np.array([[speed[0]], [speed[1]], [self.yaw_rate], [self.action['acceleration']], dx[0], dx[1]])

    def step(self, dt: float) -> None:
        self.clip_actions()
        derivative = self.derivative
        self.position += derivative[0:2, 0] * dt
        self.heading += self.yaw_rate * dt
        self.speed += self.action['acceleration'] * dt
        self.lateral_speed += derivative[4, 0] * dt
        self.yaw_rate += derivative[5, 0] * dt

        self.on_state_update()

    def clip_actions(self) -> None:
        super().clip_actions()
        # Required because of the linearisation
        self.action["steering"] = np.clip(self.action["steering"], -np.pi/2, np.pi/2)
        self.yaw_rate = np.clip(self.yaw_rate, -self.MAX_ANGULAR_SPEED, self.MAX_ANGULAR_SPEED)

    def lateral_lpv_structure(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        State: [lateral speed v, yaw rate r]

        :return: lateral dynamics A0, phi, B such that dx = (A0 + theta^T phi)x + B u
        """
        B = np.array([
            [2*self.FRICTION_FRONT / self.MASS],
            [self.FRICTION_FRONT * self.LENGTH_A / self.INERTIA_Z]
        ])

        speed_body_x = self.speed
        A0 = np.array([
            [0, -speed_body_x],
            [0, 0]
        ])

        if abs(speed_body_x) < 1:
            return A0, np.zeros((2, 2, 2)), B*0

        phi = np.array([
            [
                [-2 / (self.MASS*speed_body_x), -2*self.LENGTH_A / (self.MASS*speed_body_x)],
                [-2*self.LENGTH_A / (self.INERTIA_Z*speed_body_x), -2*self.LENGTH_A**2 / (self.INERTIA_Z*speed_body_x)]
            ], [
                [-2 / (self.MASS*speed_body_x), 2*self.LENGTH_B / (self.MASS*speed_body_x)],
                [2*self.LENGTH_B / (self.INERTIA_Z*speed_body_x), -2*self.LENGTH_B**2 / (self.INERTIA_Z*speed_body_x)]
            ],
        ])
        return A0, phi, B

    def lateral_lpv_dynamics(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        State: [lateral speed v, yaw rate r]

        :return: lateral dynamics A, B
        """
        A0, phi, B = self.lateral_lpv_structure()
        self.theta = np.array([self.FRICTION_FRONT, self.FRICTION_REAR])
        A = A0 + np.tensordot(self.theta, phi, axes=[0, 0])
        return A, B

    def full_lateral_lpv_structure(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        State: [position y, yaw psi, lateral speed v, yaw rate r]

        The system is linearized around psi = 0

        :return: lateral dynamics A, phi, B
        """
        A_lat, phi_lat, B_lat = self.lateral_lpv_structure()

        speed_body_x = self.speed
        A_top = np.array([
            [0, speed_body_x, 1, 0],
            [0, 0, 0, 1]
        ])
        A0 = np.concatenate((A_top, np.concatenate((np.zeros((2, 2)), A_lat), axis=1)))
        phi = np.array([np.concatenate((np.zeros((2, 4)), np.concatenate((np.zeros((2, 2)), phi_i), axis=1)))
                        for phi_i in phi_lat])
        B = np.concatenate((np.zeros((2, 1)), B_lat))
        return A0, phi, B

    def full_lateral_lpv_dynamics(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        State: [position y, yaw psi, lateral speed v, yaw rate r]

        The system is linearized around psi = 0

        :return: lateral dynamics A, B
        """
        A0, phi, B = self.full_lateral_lpv_structure()
        self.theta = [self.FRICTION_FRONT, self.FRICTION_REAR]
        A = A0 + np.tensordot(self.theta, phi, axes=[0, 0])
        return A, B


def simulate(dt: float = 0.1) -> None:
    import control
    time = np.arange(0, 20, dt)
    vehicle = BicycleVehicle(road=None, position=[0, 5], speed=8.3)
    xx, uu = [], []
    from highway_env.interval import LPV
    A, B = vehicle.full_lateral_lpv_dynamics()
    K = -np.asarray(control.place(A, B, -np.arange(1, 5)))
    lpv = LPV(x0=vehicle.state[[1, 2, 4, 5]].squeeze(), a0=A, da=[np.zeros(A.shape)], b=B,
              d=[[0], [0], [0], [1]], omega_i=[[0], [0]], u=None, k=K, center=None, x_i=None)

    for t in time:
        # Act
        u = K @ vehicle.state[[1, 2, 4, 5]]
        omega = 2*np.pi/20
        u_p = 0*np.array([[-20*omega*np.sin(omega*t) * dt]])
        u += u_p
        # Record
        xx.append(np.array([vehicle.position[0], vehicle.position[1], vehicle.heading])[:, np.newaxis])
        uu.append(u)
        # Interval
        lpv.set_control(u, state=vehicle.state[[1, 2, 4, 5]])
        lpv.step(dt)
        # x_i_t = lpv.change_coordinates(lpv.x_i_t, back=True, interval=True)
        # Step
        vehicle.act({"acceleration": 0, "steering": u})
        vehicle.step(dt)

    xx, uu = np.array(xx), np.array(uu)
    plot(time, xx, uu)


def plot(time: np.ndarray, xx: np.ndarray, uu: np.ndarray) -> None:
    pos_x, pos_y = xx[:, 0, 0], xx[:, 1, 0]
    psi_x, psi_y = np.cos(xx[:, 2, 0]), np.sin(xx[:, 2, 0])
    dir_x, dir_y = np.cos(xx[:, 2, 0] + uu[:, 0, 0]), np.sin(xx[:, 2, 0] + uu[:, 0, 0])
    _, ax = plt.subplots(1, 1)
    ax.plot(pos_x, pos_y, linewidth=0.5)
    dir_scale = 1/5
    ax.quiver(pos_x[::20]-0.5/dir_scale*psi_x[::20],
              pos_y[::20]-0.5/dir_scale*psi_y[::20],
              psi_x[::20], psi_y[::20],
              angles='xy', scale_units='xy', scale=dir_scale, width=0.005, headwidth=1)
    ax.quiver(pos_x[::20]+0.5/dir_scale*psi_x[::20], pos_y[::20]+0.5/dir_scale*psi_y[::20], dir_x[::20], dir_y[::20],
              angles='xy', scale_units='xy', scale=0.25, width=0.005, color='r')
    ax.axis("equal")
    ax.grid()
    plt.show()
    plt.close()


def main() -> None:
    simulate()


if __name__ == '__main__':
    main()
