import numpy as np

def T01(q):
    c1, s1 = np.cos(q), np.sin(q)
    return np.array([
        [ c1,  0.0, -s1,  0.0],
        [ s1,  0.0,  c1,  0.0],
        [ 0.0, -1.0, 0.0, 0.1625],
        [ 0.0,  0.0, 0.0,  1.0]
    ])

def T12(q):
    c2, s2 = np.cos(q), np.sin(q)
    return np.array([
        [ c2, -s2, 0.0, 0.4250*c2],
        [ s2,  c2, 0.0, 0.4250*s2],
        [ 0.0, 0.0, 1.0,  0.0],
        [ 0.0, 0.0, 0.0,  1.0]
    ])

def T23(q):
    c3, s3 = np.cos(q), np.sin(q)
    return np.array([
        [ c3, -s3, 0.0, 0.3922*c3],
        [ s3,  c3, 0.0, 0.3922*s3],
        [ 0.0, 0.0, 1.0,  0.0],
        [ 0.0, 0.0, 0.0,  1.0]
    ])

def T34(q):
    c4, s4 = np.cos(q), np.sin(q)
    return np.array([
        [ c4,  0.0, -s4,  0.0],
        [ s4,  0.0,  c4,  0.0],
        [ 0.0, -1.0, 0.0, 0.1333],
        [ 0.0,  0.0, 0.0,  1.0]
    ])

def T45(q):
    c5, s5 = np.cos(q), np.sin(q)
    return np.array([
        [ c5,  0.0,  s5,  0.0],
        [ s5,  0.0, -c5,  0.0],
        [ 0.0,  1.0,  0.0, 0.0997],
        [ 0.0,  0.0,  0.0,  1.0]
    ])

def T56(q):
    c6, s6 = np.cos(q), np.sin(q)
    return np.array([
        [ c6,  +s6,  0.0,  0.0],   
        [ s6,  -c6,  0.0,  0.0],
        [ 0.0,  0.0, -1.0,  0.0996], 
        [ 0.0,  0.0,  0.0,  1.0]
    ])

    