import time
from tkinter import Tk, Canvas, Button, Frame, LEFT
from tkinter.messagebox import askyesno
from tkinter.simpledialog import askstring

from racecar_driving.resources.bezier import Bezier
from racecar_driving.resources.util import Vector2


def world_space_to_screen_space(vector):
    return (Vector2(300, 300) + vector * 2).tuple()


def screen_space_to_world_space(x, y):
    return Vector2(x - 300, y - 300) / 2


class TrackEditor:
    def __init__(self):
        self.root = Tk()
        self.root.geometry('600x625')
        self.root.resizable(False, False)

        self.handle_radius = 7

        self.prev_x, self.prev_y = 0, 0
        self.selected_point = -1
        self.selected_line = -1
        self.moving_point = False

        self.canvas = Canvas(self.root, width=600, height=600)
        self.canvas.pack()
        self.canvas.bind('<Motion>', self.mouse_move)
        self.canvas.bind('<Button 1>', self.left_mouse_press)
        self.canvas.bind('<ButtonRelease 1>', self.left_mouse_release)
        self.canvas.bind('<Button 3>', self.right_mouse_press)

        self.bezier = None
        self.reset_track()

        self.button_frame = Frame(self.root)
        self.button_frame.pack()
        self.save_button = Button(self.button_frame, text="Save", command=self.save)
        self.load_button = Button(self.button_frame, text="Load", command=self.load)
        self.clear_button = Button(self.button_frame, text="Clear", command=self.clear)
        self.save_button.pack(side=LEFT)
        self.load_button.pack(side=LEFT)
        self.clear_button.pack(side=LEFT)

        self.render()
        self.root.mainloop()

    def mouse_move(self, event):
        if self.moving_point:
            self.move_point(event.x, event.y)
        else:
            self.handle_selection(event.x, event.y)
        self.prev_x, self.prev_y = event.x, event.y

    def left_mouse_press(self, event):
        if self.selected_point != -1:
            self.moving_point = True
        if self.selected_line != -1:
            self.bezier.split_segment(screen_space_to_world_space(event.x, event.y))
            self.selected_point = (self.selected_line + 1) * 3
            self.moving_point = True
            self.selected_line = -1

    def left_mouse_release(self, event):
        self.moving_point = False

    def right_mouse_press(self, event):
        if self.selected_point != -1 and not self.moving_point:
            self.bezier.delete_point(self.selected_point)
            self.selected_point = -1

    def move_point(self, mouse_x, mouse_y):
        self.bezier.move_point(self.selected_point, (mouse_x-self.prev_x)/2, (mouse_y-self.prev_y)/2)

    def handle_selection(self, mouse_x, mouse_y):
        self.selected_point = -1
        self.selected_line = -1

        min_dist = float('inf')
        for point_index in range(self.bezier.num_points):
            point = self.bezier.get_control_point(point_index)
            x, y = world_space_to_screen_space(point)
            distance_sq = (mouse_x - x) ** 2 + (mouse_y - y) ** 2
            if distance_sq < min_dist:
                min_dist = distance_sq
                self.selected_point = point_index
        if min_dist > self.handle_radius * self.handle_radius:
            self.selected_point = -1
        else:
            return

        min_dist = float('inf')
        for segment_index in range(self.bezier.num_segments):
            _, dist = self.bezier.get_distance_from_curve(screen_space_to_world_space(mouse_x, mouse_y), segment_index)
            if dist < min_dist:
                min_dist = dist
                self.selected_line = segment_index
        if min_dist * 2 > self.handle_radius:
            self.selected_line = -1

    def render(self):
        self.canvas.update()
        self.canvas.delete('all')

        # Draw control lines
        for segment_index in range(self.bezier.num_segments):
            for point_index in range(3):
                point0 = self.bezier.get_segment_point(segment_index, point_index)
                x0, y0 = world_space_to_screen_space(point0)
                point1 = self.bezier.get_segment_point(segment_index, point_index + 1)
                x1, y1 = world_space_to_screen_space(point1)
                self.canvas.create_line(x0, y0, x1, y1, dash=(3,))

        # Draw curve
        for segment_index in range(self.bezier.num_segments):
            t_steps = 20
            start_point = self.bezier.get_curve_point(segment_index, 0)
            prev_x, prev_y = world_space_to_screen_space(start_point)
            for step in range(t_steps):
                t = (step + 1) / t_steps
                current_point = self.bezier.get_curve_point(segment_index, t)
                cur_x, cur_y = world_space_to_screen_space(current_point)
                if self.selected_line == segment_index:
                    colour = 'yellow'
                else:
                    colour = 'blue'
                self.canvas.create_line(prev_x, prev_y, cur_x, cur_y, fill=colour, width=3)
                prev_x, prev_y = cur_x, cur_y

        # Draw control points
        for segment_index in range(self.bezier.num_segments):
            for point_index in range(3):
                point0 = self.bezier.get_segment_point(segment_index, point_index)
                x0, y0 = world_space_to_screen_space(point0)
                radius = 7
                if self.selected_point == segment_index * 3 + point_index:
                    colour = 'yellow'
                elif point_index == 0:
                    colour = 'red'
                else:
                    colour = 'black'
                self.canvas.create_oval(x0 - radius, y0 - radius, x0 + radius, y0 + radius, fill=colour)
        self.root.after(0, self.render)

    def save(self):
        name = askstring("Save", "What do you want to save this as?")
        if name:
            self.bezier.save(name)

    def load(self):
        name = askstring("Save", "What do you want to load?")
        if name:
            self.bezier = Bezier.load(name)

    def clear(self):
        if askyesno("Clear", "Are you sure you want to clear the track?"):
            self.reset_track()

    def reset_track(self):
        self.bezier = Bezier(
            Vector2(-40, -40), Vector2(-40, -20), Vector2(-40, 20), Vector2(-40, 40), Vector2(-40, 70), Vector2(40, 70),
            Vector2(40, 40), Vector2(40, 20), Vector2(40, -20), Vector2(40, -40), Vector2(40, -70), Vector2(-40, -70),
        )


if __name__ == '__main__':
    TrackEditor()
