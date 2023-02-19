from racecar_driving.resources import car


class CarGenerator:
    def __init__(self, client):
        self.client = client
        self.car = None

    def reset_car(self, position, orientation):
        if self.car is not None:
            self.car.remove()
        self.car = car.Car(self.client, position, orientation)
        return self.car
