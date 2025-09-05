import math

class LinearLR():
    def __init__(self, optimizer, epochs, initial_lr, final_lr, last_epoch=-1):
        self.optimizer = optimizer
        self.epochs = epochs
        self.initial_lr = initial_lr
        self.final_lr = final_lr
        self.last_epoch = last_epoch
        self.lr = initial_lr
        optimizer.lr = initial_lr

    def step(self, epoch):
        self.last_epoch = epoch
        self.lr = self.initial_lr + (self.final_lr - self.initial_lr) * epoch / self.epochs
        self.optimizer.lr = self.lr

    def get_lr(self):
        return self.lr


class ConstantLR():
    def __init__(self, optimizer, epochs, initial_lr, final_lr, last_epoch=-1):
        self.optimizer = optimizer
        self.epochs = epochs
        self.initial_lr = initial_lr
        self.final_lr = final_lr
        self.last_epoch = last_epoch
        self.lr = initial_lr
        optimizer.lr = initial_lr

    def step(self, epoch):
        self.last_epoch = epoch
        self.optimizer.lr = self.lr

    def get_lr(self):
        return self.lr


class CosineLR():
    def __init__(self, optimizer, epochs, initial_lr, final_lr, last_epoch=-1):
        self.optimizer = optimizer
        self.epochs = epochs
        self.initial_lr = initial_lr
        self.final_lr = final_lr
        self.last_epoch = last_epoch
        self.lr = initial_lr
        optimizer.lr = initial_lr

    def step(self, epoch):
        self.last_epoch = epoch
        self.lr = self.final_lr + 0.5 * (self.initial_lr - self.final_lr) * (1 + math.cos(math.pi * epoch / self.epochs))
        self.optimizer.lr = self.lr

    def get_lr(self):
        return self.lr