'''

Alex Rosengarten, A09956320
Mark Darmadi, A11410141
Vania Chandra, A12496582
Feb 11 2016
CSE 151 Homework 3 Problem 4

'''


import numpy as np
import matplotlib.pyplot as plot
from queue import *  

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
        

def information_gain(f, t, data):
    # IG(Z) = H(X) - H(X|Z) where H(X) is the entropy of labels 
    # distributed at a node and H(X|Z) is the entropy given the group
    # its apart of (if f <= t or not)

    N = np.shape(data)[0]
    label_freq = {}
    left  = []
    left_freq = {}
    right = []
    right_freq = {}
    for i in range(4):
        left_freq[i] = 0
        right_freq[i] = 0
        label_freq[i] = 0

    for i in range(N):
        label_freq[data[i,-1]] += 1.0
        if data[i, f] <= t:
            left.append(data[i, -1])
            left_freq[data[i,-1]] += 1.0
        else:
            right.append(data[i,-1])
            right_freq[data[i,-1]] += 1.0

    H_x = 0
    for k, v in label_freq.items():
        f = v/N
        if f != 0:
            H_x -= f*np.log(f) # calculate entropy, H(X)

        
    H_x_z0 = 0
    if len(left) != 0:
        for k, v in left_freq.items():
            f = v/len(left)
            if f != 0:
                H_x_z0 -= f*np.log(f)
    
    H_x_z1 = 0
    if len(right) != 0:
        for k, v in right_freq.items():
            f = v/len(right)
            if f != 0:
                H_x_z1 -= f*np.log(f)


    Pz0 = float(len(left))/N
    Pz1 = float(len(right))/N

    H_x_z = Pz0 * H_x_z0 + Pz1 * H_x_z1

    return H_x - H_x_z # H(X) - H(X|Z)
    
    
        
def generate_tree(data):
    root = Node(data = data, rule=(None, None))

    to_explore = Queue(maxsize=np.shape(train)[0])

    to_explore.put(root)

    count = 0
    while not to_explore.empty():

        current_node = to_explore.get()

        N, d = np.shape(current_node.data)
                
        # test if data is impure:
        if not np.all(current_node.data == current_node.data[0,:], axis=0)[-1]:
            
            max_ent = -9999999999999999999999
            f = 0
            t = 0

            for feature in range(int(np.shape(current_node.data)[1]-1)):
                for thresh in np.linspace(np.min(current_node.data[:,feature]), 
                                   np.max(current_node.data[:,feature]), 50):
                    # Calculate entropy
                    new_ent = information_gain(feature,thresh,current_node.data)
                    if new_ent > max_ent:
                        max_ent = new_ent
                        f = feature
                        t = thresh

            data_left   = np.array([[0,0,0,0,0]])
            data_right  = np.array([[0,0,0,0,0]]) 
            left = 0
            right = 0
            for i in range(np.shape(current_node.data)[0]):
                if current_node.data[i,f] <= t:
                    data_left = np.vstack((data_left, current_node.data[i,:]))
                    left += 1
                else:
                    data_right = np.vstack((data_right, current_node.data[i,:]))
                    right += 1

            print(f, t, N, max_ent) 

            if np.shape(data_left)[0] == 1:
                left_child = None
            else:
                data_left = data_left[1:,:]
                left_child = Node(data=data_left)
                print('Left: ' + str(left))

            if np.shape(data_right)[0] == 1:
                right_child = None
            else:
                data_right = data_right[1:,:]
                right_child = Node(data=data_right)
                print('Right: ' + str(right))
            
            current_node.left  = left_child
            current_node.right = right_child
            current_node.rule  = (f, t)

            if left_child is not None:
                to_explore.put(left_child)

            if right_child is not None:
                to_explore.put(right_child)

            count += 1
        else:
            current_node.label = current_node.data[0,-1]
            print('label: ' + str(N) + ' ' + str(current_node.label))

    return root

def predict(datapoint, root):
    if root.label is not None:
        return root.label
    else:
        (f, t) = root.rule
        if datapoint[f] <= t:
            return predict(datapoint, root.left)
        else:
            return predict(datapoint, root.right)

def compare_results(data, root):
    N, d = np.shape(data)

    err = 0
    for i in range(N):
        if(predict(data[i,:],root) != data[i,-1]):
            err += 1.0
    return err/N

train = None
test = None

if __name__  == "__main__":

    train = np.loadtxt('hw3train.txt')
    test  = np.loadtxt('hw3test.txt')

    # train_x = train[:, 0:-2]
    # train_y = train[:, -1]
    # test_x  = test[:, 0:-2]
    # test_y  = test[:, -1]

    root = generate_tree(train)

    print(compare_results(test, root))

    
