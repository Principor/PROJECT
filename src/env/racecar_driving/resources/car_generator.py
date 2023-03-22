import random

import numpy as np

from racecar_driving.resources import car


kwargs = [
    {
        "com_y": -1
    },
    {
        "com_y": 0.5
    }
]


class CarGenerator:

    def __init__(self, client, car_index):
        assert -1 <= car_index < len(kwargs)
        self.car_index = car_index
        self.client = client
        self.car = None

    def reset_car(self, position, orientation):
        if self.car is not None:
            self.car.remove()
        if self.car_index == -1:
            self.car = car.Car(self.client, position, orientation,
                               com_y=np.random.uniform(-1, 0.5))
        else:
            self.car = car.Car(self.client, position, orientation, **kwargs[self.car_index])
        return self.car
