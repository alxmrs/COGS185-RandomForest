import numpy as np
from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing.dummy import Value, Semaphore, Manager

def read_data(data_file_name, n_features=None, n_datapoints=-1):
    """
    Slightly Modified by Alex Rosengarten
    Source: https://github.com/cjlin1/libsvm/blob/master/python/svmutil.py
    svm_read_problem(data_file_name) -> [y, x]
    Read LIBSVM-format data from data_file_name and return labels y
    and data instances x.
    """
    prob_y = []
    prob_x = []
    i = 0
    for line in open(data_file_name):
        if i is n_datapoints:
            break
        line = line.split(None, 1)
        # In case an instance with all zero features
        if len(line) == 1: line += ['']
        label, features = line
        if n_features is None:
            xi = [0 for _ in range(len(features.split()))]
        else:
            xi = [0 for _ in range(n_features)]
        for e in features.split():
            ind, val = e.split(":")
            xi[int(ind)-1] = float(val)
        i += 1
        prob_y += [float(label)]
        prob_x += [xi]

    return prob_y, prob_x


class Data(object):
    def __init__(self, data_file_name, n_features=None, n_datapoints=-1, n_threads=None):
        self.file = data_file_name
        self.n_features = n_features
        self.n_datapoints = n_datapoints
        self.n_threads = n_threads
        # self.manager = Manager()
        # self.sema = self.manager.Semaphore()
        # self.dp_counter = self.manager.Value('i', 0)
        # self.event = self.manager.Event()

    def process_line(self, line):
        # with self.sema:
        #     if self.dp_counter.value == self.n_datapoints:
        #         self.event.set()
        #         print('event set')
        #     self.dp_counter.value += 1
        #     # print(self.dp_counter.value, self.n_datapoints)

        line = line.split(None, 1)
        # In case an instance with all zero features
        if len(line) == 1: line += ['']
        label, features = line
        if self.n_features is None:
            xi = [0.0 for _ in range(len(features.split()))]
        else:
            xi = [0.0 for _ in range(self.n_features)]
        for e in features.split():
            ind, val = e.split(":")
            xi[int(ind)-1] = float(val)

        return xi + [float(label)]

    def read_data(self):
        if self.n_threads is None:
            pool = ThreadPool()
        else:
            pool = ThreadPool(self.n_threads)

        with open(self.file) as f:
            results = pool.map(self.process_line, f)
            pool.close()
            pool.join()

        return np.array(results)
