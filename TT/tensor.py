import numpy as lb


def left_unfolding(order3tensor):
    s = order3tensor.shape
    return order3tensor.reshape(s[0]*s[1], s[2])

def right_unfolding(order3tensor):
    s = order3tensor.shape
    return order3tensor.reshape(s[0], s[1]*s[2])