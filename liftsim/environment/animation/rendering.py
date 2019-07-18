import pyglet


pyglet.resource.path = ['./environment/animation/resources']
pyglet.resource.reindex()

class Render(pyglet.window.Window):
    def __init__(self, shared):
        self.shared_mansion = shared
        self.floor_height = self.shared_mansion.attribute.FloorHeight
        self.elevator_num = self.shared_mansion.attribute.ElevatorNumber
        self.num_floor = self.shared_mansion.attribute.NumberOfFloor

        self.screen_x = int(self.elevator_num * 50 + 300)
        self.screen_y = int(self.num_floor * 12.5 * self.floor_height + 60)
        super(Render, self).__init__(width=self.screen_x, height=self.screen_y)
        self.create_window()
        

    def create_window(self):
        self.set_size(self.screen_x, self.screen_y)
        self.set_visible()

        self.load_images()
        self.init_batch()

    def center_image(self, image):
        image.anchor_x = image.width//2
        image.anchor_y = image.height//2

    def load_images(self):
        self.man_image = pyglet.resource.image('matchstick_man.png')
        self.background = pyglet.resource.image("white_bg.png")
        self.line = pyglet.resource.image("line.png")
        self.elevator = pyglet.resource.image("elevator.png")
        self.circle = pyglet.resource.image("circle.png")
        self.up = pyglet.resource.image("up.png")
        self.down = pyglet.resource.image("down.png")

        # modify images
        self.elevator.width, self.elevator.height = 43, 45
        self.center_image(self.elevator)

        self.line.width, self.line.height = self.screen_x, 5
        self.center_image(self.line)

        self.background.width, self.background.height = self.screen_x, self.screen_y
        self.center_image(self.background)
        self.background = pyglet.sprite.Sprite(img = self.background, x=self.screen_x//2, y=self.screen_y//2)

        self.man_image.width, self.man_image.height = 20, 30
        self.center_image(self.man_image)

        self.circle.width, self.circle.height = 7.5, 7.5
        self.center_image(self.circle)

        self.up.width, self.up.height = 15, 15
        self.center_image(self.up)

        self.down.width, self.down.height = 15, 15
        self.center_image(self.down)

        self.up_label = pyglet.text.Label(text="Waiting up", font_size=10, x=100, y=self.screen_y-35, anchor_x='center', color=(0,0,0,255))
        self.down_label = pyglet.text.Label(text="Waiting down", font_size=10, x=self.screen_x-100, y=self.screen_y-35, anchor_x='center', color=(0,0,0,255))
        self.level_label = pyglet.text.Label(text="Elevator Simulator", font_size=10, x=self.screen_x//2, y=self.screen_y-15, anchor_x='center', color=(0,0,0,255))

    def init_batch(self):
        '''
        line_batch: lines to separate floors, which can be initialized here as it remains unchanged
        waiting_people_batch: people on the two sides
        elevator_batch: including elevator (square) and passengers (circle)
        '''
        self.line_batch = pyglet.graphics.Batch()
        self.waiting_people_batch = pyglet.graphics.Batch()
        self.elevator_batch = list()
        self.line_ele = list()
        self.waiting_people_ele = list()
        self.elevator_ele = list()
        for i in range(self.elevator_num):
            self.elevator_batch.append(pyglet.graphics.Batch())

        for i in range(self.num_floor):
            self.line_ele.append(pyglet.sprite.Sprite(img = self.line, x=self.screen_x//2, y=12.5*self.floor_height*i+50, batch = self.line_batch))

    def update(self):
        # update waiting_people_batch
        waiting_up, waiting_down = self.shared_mansion.waiting_queue
        for ele in self.waiting_people_ele:
            ele.delete()
        self.waiting_people_ele = []
        for i in range(self.num_floor):
            # left side, waiting up people
            if len(waiting_up[i]) > 9:
                self.waiting_people_ele.append(pyglet.sprite.Sprite(img=self.man_image, x=100, y=12.5*self.floor_height*i+15, batch = self.waiting_people_batch))
                self.waiting_people_ele.append(pyglet.text.Label(text="x {}".format(len(waiting_up[i])), font_size=8, 
                            x=130, y=12.5*self.floor_height*i+15, anchor_x='center', color=(0,0,0,255), batch = self.waiting_people_batch))
            else:
                for j in range(len(waiting_up[i])):
                    self.waiting_people_ele.append(pyglet.sprite.Sprite(img = self.man_image, x=140-15*j, y=12.5*self.floor_height*i+15, batch = self.waiting_people_batch))
            # right side, waiting down people
            if len(waiting_down[i]) > 9:
                self.waiting_people_ele.append(pyglet.sprite.Sprite(img=self.man_image, x=self.screen_x-125, y=12.5*self.floor_height*i+15, batch = self.waiting_people_batch))
                self.waiting_people_ele.append(pyglet.text.Label(text="x {}".format(len(waiting_down[i])), font_size=8, 
                            x=self.screen_x-95, y=12.5*self.floor_height*i+15, anchor_x='center', color=(0,0,0,255), batch=self.waiting_people_batch))
            else:
                for j in range(len(waiting_down[i])):
                    self.waiting_people_ele.append(pyglet.sprite.Sprite(img = self.man_image, x=self.screen_x-140+15*j, y=12.5*self.floor_height*i+15, batch = self.waiting_people_batch))

        # update elevator_batch
        for ele in self.elevator_ele:
            ele.delete()
        self.elevator_ele = []
        for i in range(self.elevator_num):
            self.elevator_floor = self.shared_mansion.state.ElevatorStates[i].Floor
            self.elevator_ele.append(pyglet.sprite.Sprite(img=self.elevator, 
                            x=175+i*50, y=self.elevator_floor*self.floor_height*12.5-25, batch = self.elevator_batch[i]))
            if self.shared_mansion._elevators[i]._direction == 1:
                self.elevator_ele.append(pyglet.sprite.Sprite(img=self.up, x=158.5+i*50, y=self.elevator_floor*self.floor_height*12.5-25, batch = self.elevator_batch[i]))
            elif self.shared_mansion._elevators[i]._direction == -1:
                self.elevator_ele.append(pyglet.sprite.Sprite(img=self.down, x=158.5+i*50, y=self.elevator_floor*self.floor_height*12.5-25, batch = self.elevator_batch[i]))  
            
            # when too many passengers in an elevator, use numeric numbers to show
            if self.shared_mansion.loaded_people[i] > 9:
                self.elevator_ele.append(pyglet.text.Label(text="{}".format(self.shared_mansion.loaded_people[i]), font_size=8,
                                x=175+i*50, y=self.elevator_floor*self.floor_height*12.5-25, anchor_x='center', color=(0,0,0,255), batch = self.elevator_batch[i]))
                self.elevator_ele.append(pyglet.text.Label(text="people", font_size=7,
                                x=175+i*50, y=self.elevator_floor*self.floor_height*12.5-35, anchor_x='center', color=(0,0,0,255), batch = self.elevator_batch[i]))
            else:
                for j in range(self.shared_mansion.loaded_people[i]):
                    self.elevator_ele.append(pyglet.sprite.Sprite(img = self.circle, 
                                x=160+i*50+(j%3+1)*7.5, y=15-(j//3)*15+self.elevator_floor*self.floor_height*12.5-25, batch = self.elevator_batch[i]))
        pyglet.gl.get_current_context().set_current()

    def view(self):
        self.clear()
        self.update()

        self.dispatch_events()
        self.background.draw()
        self.level_label.draw()
        self.up_label.draw()
        self.down_label.draw()
        self.waiting_people_batch.draw()
        for i in range(self.elevator_num):
            self.elevator_batch[i].draw()
        self.line_batch.draw()

        self.flip()
