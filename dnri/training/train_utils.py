import torch
from torch.utils.tensorboard import SummaryWriter

import os


def build_scheduler(opt, params):
    lr_decay_factor = params.get('lr_decay_factor')
    lr_decay_steps = params.get('lr_decay_steps')
    if lr_decay_factor:
        return torch.optim.lr_scheduler.StepLR(opt, lr_decay_steps, lr_decay_factor)
    else:
        return None


class build_writers:
    def __init__(self, working_dir, is_test=False):
        self.writer_dir = os.path.join(working_dir, 'logs/')
        self.is_test = is_test

    def __enter__(self):
        train_writer_dir = os.path.join(self.writer_dir, 'train')
        val_writer_dir = os.path.join(self.writer_dir, 'val')
        self.train_writer = SummaryWriter(train_writer_dir)
        self.val_writer = SummaryWriter(val_writer_dir)
        if self.is_test:
            test_writer_dir = os.path.join(self.writer_dir, 'test')
            self.test_writer = SummaryWriter(test_writer_dir)
            return self.train_writer, self.val_writer, self.test_writer
        else:
            return self.train_writer, self.val_writer

    def __exit__(self, type, value, traceback):
        self.train_writer.close()
        self.val_writer.close()
        if self.is_test:
            self.test_writer.close()