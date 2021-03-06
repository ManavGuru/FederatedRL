import random
import numpy as np
import pickle

class ReplayBuffer(object):
    buffer = []
    count = 0
    def __init__(self, buffer_size, random_seed=123):
        self.buffer_size = buffer_size
        #self.count = 0
        #self.buffer = []
        random.seed(random_seed)

    def add(self, s, a, r, t, s2):
        experience = (s, a, r, t, s2)
        if self.count < self.buffer_size:
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.pop(0)
            self.buffer.append(experience)

    def size(self):
        return self.count

    def sample_batch(self, batch_size):
        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)

        s_batch = np.array([_[0] for _ in batch])
        a_batch = np.array([_[1] for _ in batch])
        r_batch = np.array([_[2] for _ in batch])
        t_batch = np.array([_[3] for _ in batch])
        s2_batch = np.array([_[4] for _ in batch])

        return s_batch, a_batch, r_batch, t_batch, s2_batch

    def clear(self):
        self.buffer = []
        self.count = 0

    def save(self):
        file = open('replay_buffer.obj', 'wb')
        pickle.dump(self.buffer, file)
        file.close()

    def load(self):
        try:
            filehandler = open('replay_buffer.obj', 'rb')
            self.buffer = pickle.load(filehandler)
            self.count = len(self.buffer)
        except:
            print('there was no file to load')

