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
        self.texts = []
        self.env = Env()
        self.agent = agent
        self.evaluation_count = 0
        self.improvement_count = 0
        self.is_moving = 0
        (self.up, self.down, self.left, self.right), self.shapes = self.load_images()
        self.canvas = self._build_canvas()
        self.text_reward(2, 2, "R : 1.0")
        self.text_reward(1, 2, "R : -1.0")
        self.text_reward(2, 1, "R : -1.0")
        self.text_reward(3, 2, "R : -1.0")

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

        self.rectangle = self._draw_shape(canvas=canvas, row=0, col=0, image=self.shapes[0])
        self._draw_shape(canvas=canvas, row=2, col=1, image=self.shapes[1])
        self._draw_shape(canvas=canvas, row=1, col=2, image=self.shapes[1])
        self._draw_shape(canvas=canvas, row=2, col=3, image=self.shapes[1])
        self._draw_shape(canvas=canvas, row=2, col=2, image=self.shapes[2])
        canvas.pack()
        return canvas

    def _draw_shape(self, canvas, row, col, image):
        x = col*UNIT+UNIT/2
        y = row*UNIT+UNIT/2
        shape = canvas.create_image(x, y, image=image)
        return shape

    def text_reward(self, row, col, content, font='Helvetica', size=10, style='normal', anchor='nw'):
        origin_x, origin_y = 5, 5
        x, y = origin_x + (UNIT * row), origin_y + (UNIT *col)
        font = (font, str(size), style)
        text = self.canvas.create_text(x, y, fill='black', text=content, font=font, anchor=anchor)
        return self.texts.append(text)

    def evaluate_policy(self):
        self.evaluation_count += 1
        for i in self.texts:
            self.canvas.delete(i)
        self.agent.policy_evaluation() #<-------
        self.print_value_table(self.agent.value_table) #------ value function

    def improve_policy(self):
        self.improvement_count += 1

    def move_by_policy(self):
        pass

    def reset(self):
        if self.is_moving == 0:
            self.evaluation_count = 0
            self.improvement_count = 0

    def print_value_table(self, value_table):
        for col in range(WIDTH):
            for row in range(HEIGHT):
                self.text_value(col, row, round(value_table[col][row], 2))

    def text_value(self, col, row, content, font='Helvetica', size=10, style='normal', anchor='nw'):
        origin_x, origin_y = 85, 70
        x, y = origin_x + (UNIT * row), origin_y + (UNIT * col)
        font = (font, str(size), style)
        text = self.canvas.create_text(x,y,fill='black', text=content, font=font, anchor=anchor)
        return self.texts.append(text)

class Env:
    def __init__(self):
        self.width = WIDTH
        self.height = HEIGHT
