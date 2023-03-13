from tkinter import Tk, Canvas

from racecar_driving.resources.bezier import Bezier
from racecar_driving.resources.util import Vector2


def world_space_to_screen_space(position):
    return position * 2 + Vector2(300, 300)


class TrackEditor:
    def __init__(self):
        self.root = Tk()
        self.root.geometry('600x600')
        self.root.resizable(False, False)

        self.canvas = Canvas(self.root, width=600, height=600)
        self.canvas.pack()

        self.bezier = Bezier(
            Vector2(-114.67, 73.08), Vector2(-131.89, 89.42), Vector2(-103.53, 115.41), Vector2(-86.17, 100.00),
            Vector2(-73.36, 88.62), Vector2(-60.55, 77.24), Vector2(-47.74, 65.87), Vector2(-28.08, 48.42),
            Vector2(-15.36, 85.73), Vector2(3.95, 75.28), Vector2(24.56, 64.11), Vector2(59.82, 7.75),
            Vector2(74.23, -3.90), Vector2(89.52, -16.26), Vector2(119.91, -8.10), Vector2(128.87, -25.71),
            Vector2(135.91, -39.54), Vector2(141.08, -87.25), Vector2(117.49, -87.50), Vector2(46.57, -88.24),
            Vector2(-24.34, -88.99),  Vector2(-95.26, -89.73), Vector2(-134.60, -90.14), Vector2(-99.41, -28.48),
            Vector2(-66.72, -46.84), Vector2(-13.31, -76.84), Vector2(13.68, -48.74), Vector2(-1.46, -34.37),
            Vector2(-39.20, 1.45), Vector2(-76.93, 37.26)
        )

        self.render()
        self.root.mainloop()

    def render(self):
        for segment_index in range(self.bezier.num_segments):
            for point_index in range(3):
                point0 = self.bezier.get_segment_point(segment_index, point_index)
                x0, y0 = world_space_to_screen_space(point0).tuple()
                point1 = self.bezier.get_segment_point(segment_index, point_index + 1)
                x1, y1 = world_space_to_screen_space(point1).tuple()

                self.canvas.create_oval(x0 - 5, y0 - 5, x0 + 5, y0 + 5, fill='black')
                self.canvas.create_line(x0, y0, x1, y1)

            t_steps = 20
            start_point = self.bezier.get_curve_point(segment_index, 0)
            prev_x, prev_y = world_space_to_screen_space(start_point).tuple()
            for step in range(t_steps):
                t = (step + 1) / t_steps
                current_point = self.bezier.get_curve_point(segment_index, t)
                cur_x, cur_y = world_space_to_screen_space(current_point).tuple()
                self.canvas.create_line(prev_x, prev_y, cur_x, cur_y)
                prev_x, prev_y = cur_x, cur_y


if __name__ == '__main__':
    TrackEditor()


