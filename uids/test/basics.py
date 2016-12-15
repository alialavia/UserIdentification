#!/usr/bin/python
import numpy as np
import struct
import sys
import numpy as np
from sklearn.decomposition import PCA
from Queue import Queue


def test_concat():
    a = []
    a.append([1,2,3])
    a.append([1,2,3])
    a.append([1,2,3])

    b = []
    b.append([1,2,3])
    b.append([1,2,3])

    all2 = np.array([])

    all2 = np.concatenate((all2, a)) if all2.size else a
    all2 = np.concatenate((all2, b))

    print all2


def test_type_conversion():
    print type(np.asscalar(np.int64(1)))


def test_dict_properties():
    nice_names = {}

    user_id = 2
    user_name = "hans muster"

    nice_names[int(user_id)] = user_name

    print nice_names[int(user_id)]

def test_np_arrays():
    y = np.array([[3,3,4], [4,4,6], [1,1,1], [2,2,2]])
    print np.shape(y)
    print "Axis dimension: 0: {}, 1: {}".format(np.size(y,0), np.size(y,1))

    tot_elems = np.size(y,0)
    nr_elems = 4
    start_index = tot_elems-nr_elems
    print y[start_index:tot_elems,:]


def test_pca():

    reduce_dim_to = 3

    X = np.array([[-1, -1,3], [-2, -1,2], [-3, -2,1], [1, 1,7], [2, 1,9], [3, 2,2]])
    print "PCA data: {} samples, {} dim".format(np.size(X,0), np.size(X,1))
    pca = PCA(n_components=reduce_dim_to)
    pca.fit(X)

    # ---------------------------------
    vector = np.array([[-1,-1,3]])
    print "Input data: {} samples, {} dim".format(np.size(vector,0), np.size(vector,1))
    print np.shape(vector)
    transformed = pca.transform(vector)
    print np.shape(transformed)
    # ---------------------------------

    vector = np.array([[-1, -1, 3], [-1, -1, 3]])
    print "Input data: {} samples, {} dim".format(np.size(vector, 0), np.size(vector, 1))
    transformed = pca.transform(vector)
    print np.shape(transformed)


def test_dict_reduct():
    target_class = 1

    mydict = {}
    mydict[3] = [-1,1,-1]
    mydict[5] = [1,1,-1]
    mydict[1] = [-1,-1,-1]
    mydict[1] = [-1,-1,-1]

    vals = np.array(mydict.values())

    print vals
    # print vals[:, 0]
    # print vals[vals, 0]


    # counts = [1 for res in np.array(mydict.values()).T if len(res[res > 0]) > 1]

    for id, col in zip(mydict.keys(), np.array(mydict.values()).T):
        print id
        print col

    # only 1x1 and rest -1 or all -1
    # = nr elements =1 not greater than 1
    # or wrong class determined = not target class selected

