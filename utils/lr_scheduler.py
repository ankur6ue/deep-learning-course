import math

class LinearLR():
    def __init__(self, optimizer, num_training_steps, initial_lr, final_lr, num_warmup_steps = 0, last_step=-1):
        self.optimizer = optimizer
        self.num_training_steps = num_training_steps
        self.initial_lr = initial_lr
        self.final_lr = final_lr
        self.last_step = last_step
        self.lr = initial_lr
        self.num_warmup_steps = num_warmup_steps
        self.step(0) # if num_warmup_steps == 0, stepping will set self.ir = initial_lr, else to 0
        optimizer.lr = self.lr

    def step(self, curr_step):
        if curr_step < self.num_warmup_steps:
            self.lr = self.initial_lr * curr_step / self.num_warmup_steps
        else:
            self.lr = self.initial_lr + (self.final_lr - self.initial_lr) * (curr_step - self.num_warmup_steps) / (
                    self.num_training_steps - self.num_warmup_steps)
        self.optimizer.lr = self.lr

    def get_lr(self):
        return self.lr


class ConstantLR():
    def __init__(self, optimizer, num_training_steps, initial_lr, final_lr, num_warmup_steps=0, last_step=-1):
        self.optimizer = optimizer
        self.num_training_steps = num_training_steps
        self.initial_lr = initial_lr
        self.final_lr = final_lr
        self.last_step = last_step
        self.lr = initial_lr
        self.num_warmup_steps = num_warmup_steps
        optimizer.lr = initial_lr

    def step(self, curr_step):
        self.optimizer.lr = self.lr

    def get_lr(self):
        return self.lr


class CosineLR():
    def __init__(self, optimizer, num_training_steps, initial_lr, final_lr, num_warmup_steps=0, last_step=-1):
        self.optimizer = optimizer
        self.num_training_steps = num_training_steps
        self.initial_lr = initial_lr
        self.final_lr = final_lr
        self.last_step = last_step
        self.lr = initial_lr
        self.num_warmup_steps = num_warmup_steps
        optimizer.lr = initial_lr

    def step(self, curr_step):
        if curr_step < self.num_warmup_steps:
            self.lr = self.initial_lr * curr_step / self.num_warmup_steps
        else:
            self.lr = self.final_lr + 0.5 * (self.initial_lr - self.final_lr) * (1 + math.cos(math.pi * (curr_step - self.num_warmup_steps) /
                                                                            (self.num_training_steps - self.num_warmup_steps)))

        self.optimizer.lr = self.lr

    def get_lr(self):
        return self.lr