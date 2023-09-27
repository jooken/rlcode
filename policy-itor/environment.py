import tkinter as tk
from PIL import Image
from PIL.ImageTk import PhotoImage


UNIT = 100  # Pixel Count of a cell
WIDTH = 5 # Horizontal Cell Count on Grid World
HEIGHT = 5 # Vertical Cell Count on Grid World


class GraphicDisplay(tk.Tk):
    def __init__(self, agent):
        super(GraphicDisplay, self).__init__()
        self.title('Policy Iteration')
        self.geometry('{0}x{1}'.format(WIDTH * UNIT, HEIGHT * UNIT + 50))
        self.env = Env()
        self.agent = agent
        (self.up, self.down, self.left, self.right), self.shapes = self.load_images()
        self.canvas = self._build_canvas()
        self.text_reward(2, 2, "R : 1.0")
        self.text_reward(1, 2, "R : -1.0")
        self.text_reward(2, 1, "R : -1.0")

    def load_images(self):
        up = PhotoImage(Image.open("../img/up.png").resize((13,13)))
        down = PhotoImage(Image.open("../img/down.png").resize((13,13)))
        left = PhotoImage(Image.open("../img/left.png").resize((13,13)))
        right = PhotoImage(Image.open("../img/right.png").resize((13,13)))
        rectangle = PhotoImage(Image.open("../img/rectangle.png").resize((65,65)))
        triangle = PhotoImage(Image.open("../img/triangle.png").resize((65,65)))
        circle = PhotoImage(Image.open("../img/circle.png").resize((65,65)))
        return (up, down, left, right), (rectangle, triangle, circle)

    def _build_canvas(self):
        canvas = tk.Canvas(self, bg='white', height=HEIGHT*UNIT, width=WIDTH*UNIT)

        evaluate_button = tk.Button(self, text="Evaluate", command=self.evaluate_policy)
        evaluate_button.configure(width=10, activebackground="#33B5E5")
        canvas.create_window(WIDTH*UNIT*0.13, HEIGHT*UNIT+10, window=evaluate_button)

        improve_button = tk.Button(self, text="Improve", command=self.improve_policy)
        improve_button.configure(width=10, activebackground="#33B5E5")
        canvas.create_window(WIDTH*UNIT*0.37, HEIGHT*UNIT+10, window=improve_button)

        move_by_policy_button = tk.Button(self, text="Move", command=self.move_by_policy)
        move_by_policy_button.configure(width=10, activebackground="#33B5E5")
        canvas.create_window(WIDTH*UNIT*0.62, HEIGHT*UNIT+10, window=move_by_policy_button)

        reset_button = tk.Button(self, text="Reset", command=self.reset)
        reset_button.configure(width=10, activebackground="#33B5E5")
        canvas.create_window(WIDTH*UNIT*0.87, HEIGHT*UNIT+10, window=reset_button)

        for col in range(0, WIDTH*UNIT, UNIT):
            x0, y0, x1, y1 = col, 0, col, HEIGHT*UNIT
            canvas.create_line(x0, y0, x1, y1)

        for row in range(0, HEIGHT*UNIT, UNIT):
            x0, y0, x1, y1 = 0, row, WIDTH*UNIT, row
            canvas.create_line(x0, y0, x1, y1)

        self.rectangle = canvas.create_image(50,50,image=self.shapes[0])
        canvas.create_image(250, 150, image=self.shapes[1])
        canvas.create_image(150, 250, image=self.shapes[1])
        canvas.create_image(250, 250, image=self.shapes[2])

        canvas.pack()
        return canvas

    def text_reward(self, x, y, s):
        pass

    def evaluate_policy(self):
        pass

    def improve_policy(self):
        pass

    def move_by_policy(self):
        pass

    def reset(self):
        pass

class Env:
    pass