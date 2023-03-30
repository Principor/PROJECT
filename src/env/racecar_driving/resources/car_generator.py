import random

import numpy as np

from racecar_driving.resources import car


"""
List of settings for pre-designed cars
"""
kwargs = [
    {
        "com_y": -1
    },
    {
        "com_y": 0.5
    }
]


class CarGenerator:
    """
    Generate new cars for the start of each episode

    :param client: The physics client for the car
    :param car_index: The index of the car to generate
    """
    def __init__(self, client, car_index):
        assert -1 <= car_index < len(kwargs)
        self.car_index = car_index
        self.client = client
        self.car = None

    def reset_car(self, position, orientation):
        """
        Remove old car and generate a new one

        :param position: The position to generate the car at
        :param orientation: The orientation to generate the car with
        :return: The new car
        """
        if self.car is not None:
            self.car.remove()

        if self.car_index == -1:
            # Generate random car
            self.car = car.Car(self.client, position, orientation,
                               com_y=np.random.uniform(-1, 0.5))
        else:
            # Load pre-designed car
            self.car = car.Car(self.client, position, orientation, **kwargs[self.car_index])
        return self.car
