import numpy as np
import matplotlib.pyplot as plot
from queue import *
from multiprocessing.dummy import Pool as ThreadPool

DEBUG = True

class Node(object):
    def __init__(self, left=None, right=None, data=None, rule=(None, None), label=None):
        self.left = left
        self.right = right
        self.data = data
        self.rule = rule
        self.label = label

    def copy(self, node):
        self.left = node.left
        self.right = node.right
        self.data = node.data
        self.rule = node.rule


class ID3(object):
    def __init__(self, train, linspace=100, n_features=-1):
        self.train = train
        self.linspace = linspace
        self.num_features = n_features
        self.root = self.generate_tree(self.train)

    def information_gain(self, f, t, data):
        '''
        Calculates the decrease in conditional entropy given a condition Z: IG(Z) = H(X) - H(X|Z) where H(X) is the
        entropy of labels distributed at a node and H(X|Z) is the entropy given the group its apart of (if f <= t or not)
        :param f: feature
        :param t: threshold
        :param data: data
        :return: information gain
        '''

        N, d = np.shape(data)
        label_freq = {}
        left = []
        left_freq = {}
        right = []
        right_freq = {}

        num_classes = np.max(data[:, -1])
        # print(int(num_classes+1))

        for i in range(int(num_classes+1)):
            left_freq[i] = 0
            right_freq[i] = 0
            label_freq[i] = 0

        for i in range(N):
            label_freq[data[i, -1]] += 1.0

            if data[i, f] <= t:
                left.append(data[i, -1])
                left_freq[data[i, -1]] += 1.0
            else:
                right.append(data[i,-1])
                right_freq[data[i, -1]] += 1.0

        # Calculate entropy, H(X)
        H_x = 0
        for k, v in label_freq.items():
            f = v/N
            if f != 0:
                H_x -= f*np.log(f)

        # Calculate H(X | Z = 0)
        H_x_z0 = 0
        if len(left) != 0:
            for k, v in left_freq.items():
                f = v/len(left)
                if f != 0:
                    H_x_z0 -= f*np.log(f)

        # Calculate H(X | Z = 1)
        H_x_z1 = 0
        if len(right) != 0:
            for k, v in right_freq.items():
                f = v/len(right)
                if f != 0:
                    H_x_z1 -= f*np.log(f)

        # Calculate H(X | Z)
        Pz0 = float(len(left))/N
        Pz1 = float(len(right))/N
        H_x_z = Pz0 * H_x_z0 + Pz1 * H_x_z1

        # Return IG: H(X) - H(X|Z)
        return H_x - H_x_z

    def generate_tree(self, data):
        self.root = Node(data=data, rule=(None, None))

        to_explore = Queue(maxsize=np.shape(data)[0])

        to_explore.put(self.root)

        count = 0
        while not to_explore.empty():

            current_node = to_explore.get()

            N, d = np.shape(current_node.data)

            # Bag features or use all the features
            if self.num_features == -1:
                feature_space = range(d-1)
            else:
                feature_space = np.random.choice(d-1, size=self.num_features, replace=False)

            # test if data is impure:
            if not np.all(current_node.data == current_node.data[0, :], axis=0)[-1]:

                max_ent = -9999999999999999999999
                feature = 0
                thr_i = 0   # Initial threshold
                thr_f = 0   # Final threshold

                for f in feature_space:
                    f_min = np.min(current_node.data[:, f])
                    f_max = np.max(current_node.data[:, f])
                    for t in np.linspace(f_min, f_max, self.linspace):

                        # Calculate entropy
                        new_ent = self.information_gain(f, t, current_node.data)

                        # Find first acceptable threshold and feature
                        if new_ent > max_ent:
                            max_ent = new_ent
                            feature = f
                            thr_i = t
                        # Find last acceptable threshold
                        # elif new_ent == max_ent:
                        #     # feature = f
                        #     thr_f = t

                # Calculate midpoint of range of acceptable thresholds, if applicable
                if thr_f >= thr_i:
                    thresh = (thr_f+thr_i)/2
                else:
                    thresh = thr_i

                data_left = np.zeros((1, d))
                data_right = np.zeros((1, d))
                left = 0
                right = 0
                for i in range(np.shape(current_node.data)[0]):
                    if current_node.data[i, feature] <= thresh:
                        data_left = np.vstack((data_left, current_node.data[i, :]))
                        left += 1
                    else:
                        data_right = np.vstack((data_right, current_node.data[i, :]))
                        right += 1

#                 print(feature, thresh, N, max_ent)

                if np.shape(data_left)[0] == 1:
                    left_child = None
                else:
                    data_left = data_left[1:, :]
                    left_child = Node(data=data_left)
#                     print('Left: ' + str(left))

                if np.shape(data_right)[0] == 1:
                    right_child = None
                else:
                    data_right = data_right[1:, :]
                    right_child = Node(data=data_right)
#                     print('Right: ' + str(right))

                current_node.left = left_child
                current_node.right = right_child
                current_node.rule = (feature, thresh)

                if left_child is not None:
                    to_explore.put(left_child)

                if right_child is not None:
                    to_explore.put(right_child)

                count += 1
            else:
                current_node.label = current_node.data[0, -1]
#                 print('label: ' + str(N) + ' ' + str(current_node.label))

        return self.root

    def parse(self, data_point, root):
        if root.label is not None:
            return root.label
        else:
            (f, t) = root.rule
            if data_point[f] <= t:
                if root.left:
                    return self.parse(data_point, root.left)
            else:
                if root.right:
                    return self.parse(data_point, root.right)

    def predict(self, data_point):
        return self.parse(data_point, self.root)

    def error_rate(self, test):
        N, d = np.shape(test)

        err = 0
        for i in range(N):
            if self.predict(test[i, :-1]) != test[i, -1]:
                err += 1.0
        return err/N


def svm_read_problem(data_file_name, num_features=None):
    """
    Slightly Modified by Alex Rosengarten
    Source: https://github.com/cjlin1/libsvm/blob/master/python/svmutil.py
    svm_read_problem(data_file_name) -> [y, x]
    Read LIBSVM-format data from data_file_name and return labels y
    and data instances x.
    """
    prob_y = []
    prob_x = []
    for line in open(data_file_name):
        line = line.split(None, 1)
        # In case an instance with all zero features
        if len(line) == 1: line += ['']
        label, features = line
        if num_features is None:
            xi = [0 for _ in range(len(features.split()))]
        else:
            xi = [0 for _ in range(num_features)]
        for e in features.split():
            ind, val = e.split(":")
            xi[int(ind)-1] = float(val)
        prob_y += [float(label)]
        prob_x += [xi]

    return prob_y, prob_x

class RandomForest(object):
    def __init__(self, train, trees=100, subsample=.10, linspace=100, n_features=None, n_threads=None):
        self.training_set = train
        self.N, self.D = self.training_set.shape
        self.num_trees = trees
        self.subsample_size = int(self.N*subsample)
        self.linspace = linspace
        if n_features is None:
            self.num_features = int(np.sqrt(self.D))
        else:
            self.num_features = n_features

        self.subsamples = self.bootstrap()
        self.forest = self.generate_forest(n_threads)

    def bootstrap(self):
        inds = [np.random.choice(self.N, self.subsample_size) for _ in range(self.num_trees)]
        return [np.array([self.training_set[i] for i in ind]) for ind in inds]

    # def generate_forest(self):
    #     trees = []
    #     for i in range(self.num_trees):
    #         trees.append(ID3(self.subsamples[i],  self.linspace, n_features=self.num_features))
    #         print(str(i) + ' of ' + str(self.num_trees) + ' complete.')
    #     return trees

    def create_tree(self, sample):
        return ID3(sample, linspace=self.linspace, n_features=self.num_features)

    def generate_forest(self, n_threads=None):
        if n_threads is None:
            pool = ThreadPool()
        else:
            pool = ThreadPool(n_threads)

        forest = pool.map(self.create_tree, self.subsamples)
        pool.close()
        pool.join()
        return forest

    def predict(self, data_point):
        labels = {}
        if self.forest is None:
            return None

        for tree in self.forest:
            p = tree.predict(data_point)

            if p not in labels:
                labels[p] = 0.0

            labels[p] += 1.0

        return max(labels.keys(), key=(lambda key: labels[key]))  # return label with highest frequency in dictionary

    def error_rate(self, test):
        N, d = np.shape(test)

        err = 0
        for i in range(N):
            if self.predict(test[i, :-1]) != test[i, -1]:
                err += 1.0
        return err/N


if __name__ == '__main__':
    # print('Reading training and testing data...')
    # covtype_y, covtype_x = svm_read_problem('covtype.scale01', 54)
    # # poker_t_y, poker_t_x = svm_read_problem('poker.t')
    # print('Data loaded. Prepping Data...')

    print('Reading training and testing data...')
    poker_y, poker_x = svm_read_problem('poker')
    poker_t_y, poker_t_x = svm_read_problem('poker.t')
    print('Data loaded.')

    poker_x = np.array(poker_x)
    poker_y = np.array([poker_y]).T
    poker_train = np.concatenate((poker_x, poker_y), 1)

    poker_t_x = np.array(poker_t_x)
    poker_t_y = np.array([poker_t_y]).T
    poker_test = np.concatenate((poker_t_x, poker_t_y), 1)
    print('Data prepped.')

    N, d = poker_train.shape

    train_set_ind = set(np.random.choice(N, int(N * 0.8), replace=False))
    all_ind = set(range(N))
    test_set_ind = all_ind.difference(train_set_ind)

    poker_train_sample = [poker_train[i, :] for i in train_set_ind]
    poker_train_sample = np.array(poker_train_sample)

    poker_test_sample = [poker_train[i, :] for i in test_set_ind]
    poker_test_sample = np.array(poker_test_sample)


    print('Generating random forest')
    forest = RandomForest(poker_train_sample, 1, 0.10, 15, n_features=3)
    print('Forest generated. Now calculating test error...')

    print('test error: ' + str(forest.error_rate(poker_test_sample)))