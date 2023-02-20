from racecar_driving.resources import car

"""
Car configurations sampled uniformly from following ranges
Mass: 75 - 125kg
Wheelbase: 2.5 - 3.5m
Track width: 2.1 - 3m
Center of Mass y-axis Offset: -0.5 - 0m
Center of Mass z-axis Offset: -1 - 0.5m
"""
kwargs = [
    {
        "mass": 103,
        "wheelbase": 2.6,
        "track_width": 2.1,
        "com_y": -0.3,
        "com_z": 0.1,
     },
    {
        "mass": 113,
        "wheelbase": 3.5,
        "track_width": 2.9,
        "com_y": -0.1,
        "com_z": -0.8,
    },
    {
        "mass": 106,
        "wheelbase": 2.7,
        "track_width": 2.3,
        "com_y": -0.4,
        "com_z": -0.6,
    },
    {
        "mass": 75,
        "wheelbase": 3.2,
        "track_width": 2.8,
        "com_y": 0,
        "com_z": 0.5,
    }
]


class CarGenerator:

    def __init__(self, client, car_index):
        assert 0 <= car_index < len(kwargs)
        self.car_index = car_index
        self.client = client
        self.car = None

    def reset_car(self, position, orientation):
        if self.car is not None:
            self.car.remove()
        self.car = car.Car(self.client, position, orientation, **kwargs[self.car_index])
        return self.car
