from tkinter import *
from random import randint
from NEAT.neat import *
from math import sqrt

# constants that go in the making of the grid used for the snake's movment
GRADUATION = 40
PIXEL = 10
STEP = 2 * PIXEL
WD = PIXEL * GRADUATION
HT = PIXEL * GRADUATION
# constants that go into specifying the shapes' sizes
OB_SIZE_FACTOR = 1
SN_SIZE_FACTOR = 0.9
OB_SIZE = PIXEL * OB_SIZE_FACTOR
SN_SIZE = PIXEL * SN_SIZE_FACTOR
# color constants
BG_COLOR = 'black'
OB_COLOR = 'red'
SN_COLOR = 'white'
# a dictionary to ease access to a shape's type in the Shape class
SN = 'snake'
OB = 'obstacle'
SIZE = {SN: SN_SIZE, OB: OB_SIZE}
# constants for keyboard input
UP = 'Up'
DOWN = 'Down'
RIGHT = 'Right'
LEFT = 'Left'
# a dictionary to ease access to 'directions'
DIRECTIONS = {0: [0, -1], 1: [0, 1], 2: [1, 0], 3: [-1, 0]}
AXES = {0: 'Vertical', 1: 'Vertical', 2: 'Horizontal', 3: 'Horizontal'}
DIRECTION_CONVERT = {0: UP, 1: DOWN, 2: RIGHT, 3: LEFT}
# refresh time for the perpetual motion
REFRESH_TIME = 10


class Master(Canvas):
    """create the game canvas, the snake, the obstacle, keep track of the score"""
    def __init__(self, boss=None):
        super().__init__(boss)
        self.configure(width=WD, height=HT, bg=BG_COLOR)
        self.running = 0
        self.snake = None
        self.obstacle = None
        self.direction = None
        self.current = None
        self.score = Scores(boss)
        self.neat = Neat(input_size=3 * 3, output_size=4)
        self.neat.load()
        self.generation = StringVar(self, '0')
        self.species = StringVar(self, '0')
        self.network = StringVar(self, '0')
        self.fitness = StringVar(self, '0')
        self.max_fitness = StringVar(self, '0')

    def start(self):
        """start snake game"""
        if self.running == 0:
            self.snake = Snake(self)
            self.obstacle = Obstacle(self)
            self.direction = RIGHT
            self.current = Movement(self, randint(0, 3))
            self.current.begin()
            self.running = 1
            self.generation.set(self.neat.generation())
            self.species.set(self.neat.current_species())
            self.network.set(self.neat.current_network())
            self.fitness.set('0')

    def clean(self):
        """restarting the game"""
        if self.running == 1:
            self.score.reset()
            self.current.stop()
            self.running = 0
            self.obstacle.delete()
            for block in self.snake.blocks:
                block.delete()

            self.neat.next()

    def redirect(self, event):
        """taking keyboard inputs and moving the snake accordingly"""
        if 1 == self.running and \
                event.keysym in AXES.keys() and\
                AXES[event.keysym] != AXES[self.direction]:
            self.current.flag = 0
            self.direction = event.keysym
            self.current = Movement(self, event.keysym)  # a new instance at each turn to avoid confusion/tricking
            self.current.begin()  # program gets tricked if the user presses two arrow keys really quickly

    '''
        -1: out of map
        -1: snake body
        1: empty space
        2: apple
    '''
    def map(self):
        tile = [1 for _ in range(STEP * STEP)]

        for block in self.snake.blocks:
            x = (block.x - 10) // STEP
            y = (block.y - 10) // STEP
            tile[y * STEP + x] = -1

        x = (self.obstacle.x - 10) // STEP
        y = (self.obstacle.y - 10) // STEP
        tile[y * STEP + x] = 2

        return tile

    # get 3x3 sight around given x, y
    def sight(self, x, y):
        sight = [1 for _ in range(3 * 3)]

        for dx in range(-1, 2):
            for dy in range(-1, 2):
                xx = x + dx
                yy = y + dy

                if xx < 0 or xx >= STEP or yy < 0 or yy >= STEP:
                    sight[(dy + 1) * 3 + (dx + 1)] = -1
                else:
                    # check if we see snake body
                    for block in self.snake.blocks:
                        snake_x = (block.x - 10) // STEP
                        snake_y = (block.y - 10) // STEP

                        if xx == snake_x and yy == snake_y:
                            sight[(dy + 1) * 3 + (dx + 1)] = -1

                    # check if we see food
                    food_x = (self.obstacle.x - 10) // STEP
                    food_y = (self.obstacle.y - 10) // STEP

                    if xx == food_x and yy == food_y:
                        sight[(dy + 1) * 3 + (dx + 1)] = 2

        return sight

class Scores:
    """Objects that keep track of the score and high score"""
    def __init__(self, boss=None):
        self.counter = StringVar(boss, '0')
        self.maximum = StringVar(boss, '0')

    def increment(self):
        score = int(self.counter.get()) + 1
        maximum = max(score, int(self.maximum.get()))
        self.counter.set(str(score))
        self.maximum.set(str(maximum))

    def reset(self):
        self.counter.set('0')


class Shape:
    """This is a template to make obstacles and snake body parts"""
    def __init__(self, can, a, b, kind):
        self.can = can
        self.x, self.y = a, b
        self.kind = kind
        if kind == SN:
            self.ref = Canvas.create_rectangle(self.can,
                                               a - SN_SIZE, b - SN_SIZE,
                                               a + SN_SIZE, b + SN_SIZE,
                                               fill=SN_COLOR,
                                               width=2)
        elif kind == OB:
            self.ref = Canvas.create_oval(self.can,
                                          a - OB_SIZE, b - OB_SIZE,
                                          a + SN_SIZE, b + SN_SIZE,
                                          fill=OB_COLOR,
                                          width=2)

    def modify(self, a, b):
        self.x, self.y = a, b
        self.can.coords(self.ref,
                        a - SIZE[self.kind], b - SIZE[self.kind],
                        a + SIZE[self.kind], b + SIZE[self.kind])

    def delete(self):
        self.can.delete(self.ref)


class Obstacle(Shape):
    """snake food"""
    def __init__(self, can):
        """only create the obstacles where there is no snake body part"""
        self.can = can
        p = int(GRADUATION/2 - 1)
        n, m = randint(0, p), randint(0, p)
        a, b = PIXEL * (2 * n + 1), PIXEL * (2 * m + 1)
        while [a, b] in [[block.x, block.y] for block in self.can.snake.blocks]:
            n, m = randint(0, p), randint(0, p)
            a, b = PIXEL * (2 * n + 1), PIXEL * (2 * m + 1)
        super().__init__(can, a, b, OB)


class Block(Shape):
    """snake body part"""
    def __init__(self, can, a, y):
        super().__init__(can, a, y, SN)


class Snake:
    """a snake keeps track of its body parts"""
    def __init__(self, can):
        """initial position chosen by me"""
        self.can = can
        a = PIXEL + 2 * int(GRADUATION/4) * PIXEL
        self.blocks = [Block(can, a, a)]

    def move(self, path):
        """an elementary step consisting of putting the tail of the snake in the first position"""
        old_x = self.blocks[-1].x
        old_y = self.blocks[-1].y
        new_x = old_x + STEP * path[0]# % WD
        new_y = old_y + STEP * path[1]# % HT

        if new_x < 0 or new_x > WD or new_y < 0 or new_y > HT:
            self.can.clean()
            self.can.start()
        elif new_x == self.can.obstacle.x and new_y == self.can.obstacle.y:  # check if we find food
            self.can.score.increment()
            self.can.obstacle.delete()
            self.blocks.append(Block(self.can, new_x, new_y))
            self.can.obstacle = Obstacle(self.can)
            # +100 pts for finding food
            fitness = self.can.neat.add_fitness(100)
            self.can.fitness.set(fitness)
            if fitness > int(self.can.max_fitness.get()):
                self.can.max_fitness.set(fitness)

        elif [new_x, new_y] in [[block.x, block.y] for block in self.blocks]:  # check if we hit a body part
            self.can.clean()
            self.can.start()
        else:
            def distance(x1, y1, x2, y2):
                return sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

            old_dist = distance(old_x, old_y, self.can.obstacle.x, self.can.obstacle.y)
            new_dist = distance(new_x, new_y, self.can.obstacle.x, self.can.obstacle.y)

            # got further from food => penalty
            if old_dist < new_dist:
                fitness = self.can.neat.add_fitness(-2)
            else:
                fitness = self.can.neat.add_fitness(1)

            self.can.fitness.set(fitness)
            if fitness > int(self.can.max_fitness.get()):
                self.can.max_fitness.set(fitness)

            # too bad fitness
            if fitness < -150:
                self.can.clean()
                self.can.start()

            self.blocks[0].modify(new_x, new_y)
            self.blocks = self.blocks[1:] + [self.blocks[0]]


class Movement:
    """object that enters the snake into a perpetual state of motion in a predefined direction"""
    def __init__(self, can, direction):
        self.flag = 1
        self.can = can
        self.direction = direction

    def begin(self):
        """start the perpetual motion"""
        if self.flag > 0:
            #map = self.can.map()
            dx, dy = DIRECTIONS[self.direction]
            x = (self.can.snake.blocks[-1].x - 10) // STEP
            y = (self.can.snake.blocks[-1].y - 10) // STEP
            sight = self.can.sight(x + dx, y + dy)
            direction = self.can.neat.evaluate(sight)

            if self.direction != direction and \
                direction in AXES.keys() and \
                AXES[direction] != AXES[self.direction]:
                self.direction = direction

            self.can.snake.move(DIRECTIONS[self.direction])
            self.can.after(REFRESH_TIME, self.begin)

    def stop(self):
        """stop the perpetual movement"""
        self.flag = 0

root = Tk()
root.title("Snake Game")
game = Master(root)
game.grid(column=1, row=0, rowspan=3)
#root.bind("<Key>", game.redirect)
buttons = Frame(root, width=35, height=2*HT/5)
Button(buttons, text='Start', command=game.start).grid()
Button(buttons, text='Stop', command=game.clean).grid()
Button(buttons, text='Quit', command=root.destroy).grid()
buttons.grid(column=0, row=0)
scoreboard = Frame(root, width=35, height=3*HT/5)
Label(scoreboard, text='Generation').grid()
Label(scoreboard, textvariable=game.generation).grid()
Label(scoreboard, text='Species').grid()
Label(scoreboard, textvariable=game.species).grid()
Label(scoreboard, text='Network').grid()
Label(scoreboard, textvariable=game.network).grid()
Label(scoreboard, text='Fitness').grid()
Label(scoreboard, textvariable=game.fitness).grid()
Label(scoreboard, text='Max Fitness').grid()
Label(scoreboard, textvariable=game.max_fitness).grid()
Label(scoreboard, text='High Score').grid()
Label(scoreboard, textvariable=game.score.maximum).grid()
scoreboard.grid(column=0, row=2)

root.mainloop()
