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


def test_cont_samples(predictions, target_class):
    nr_cont_samples = 0
    for class_id, col in predictions.iteritems():
        nr_dects = len(col[col > 0])
        # if wrongly predicted: class is -1 or w
        nr_samples = len(col)
        if class_id == target_class:
            nr_cont_samples += (nr_samples - nr_dects)
        else:
            nr_cont_samples += nr_dects
    return nr_cont_samples


def test_train_error():
    # predictions = {2:np.array([-1,-1,-1,1,1,-1,-1]), 1:np.array([-1,-1,-1,-1,-1,-1,-1])}
    # predictions = {2:np.array([-1,-1,-1,1,1,-1,-1])}
    # print test_cont_samples(predictions)
    classifiers = [('a',1), ('b',2), ('c',3)]
    nr_batches = 4
    total_error_train = [None] * len(classifiers)
    # total_error_train[classifier_nr][batch_nr]
    for i in range(1, nr_batches):
        j = 0
        for clf_name, clf in classifiers:
            if i == 1:
                total_error_train[j] = []
            total_error_train[j].append(2)
            j += 1
        print total_error_train


def test_bb():
    TEMPLATE = np.float32([
        (0.0792396913815, 0.339223741112), (0.0829219487236, 0.456955367943),
        (0.0967927109165, 0.575648016728), (0.122141515615, 0.691921601066),
        (0.168687863544, 0.800341263616), (0.239789390707, 0.895732504778),
        (0.325662452515, 0.977068762493), (0.422318282013, 1.04329000149),
        (0.531777802068, 1.06080371126), (0.641296298053, 1.03981924107),
        (0.738105872266, 0.972268833998), (0.824444363295, 0.889624082279),
        (0.894792677532, 0.792494155836), (0.939395486253, 0.681546643421),
        (0.96111933829, 0.562238253072), (0.970579841181, 0.441758925744),
        (0.971193274221, 0.322118743967), (0.163846223133, 0.249151738053),
        (0.21780354657, 0.204255863861), (0.291299351124, 0.192367318323),
        (0.367460241458, 0.203582210627), (0.4392945113, 0.233135599851),
        (0.586445962425, 0.228141644834), (0.660152671635, 0.195923841854),
        (0.737466449096, 0.182360984545), (0.813236546239, 0.192828009114),
        (0.8707571886, 0.235293377042), (0.51534533827, 0.31863546193),
        (0.516221448289, 0.396200446263), (0.517118861835, 0.473797687758),
        (0.51816430343, 0.553157797772), (0.433701156035, 0.604054457668),
        (0.475501237769, 0.62076344024), (0.520712933176, 0.634268222208),
        (0.565874114041, 0.618796581487), (0.607054002672, 0.60157671656),
        (0.252418718401, 0.331052263829), (0.298663015648, 0.302646354002),
        (0.355749724218, 0.303020650651), (0.403718978315, 0.33867711083),
        (0.352507175597, 0.349987615384), (0.296791759886, 0.350478978225),
        (0.631326076346, 0.334136672344), (0.679073381078, 0.29645404267),
        (0.73597236153, 0.294721285802), (0.782865376271, 0.321305281656),
        (0.740312274764, 0.341849376713), (0.68499850091, 0.343734332172),
        (0.353167761422, 0.746189164237), (0.414587777921, 0.719053835073),
        (0.477677654595, 0.706835892494), (0.522732900812, 0.717092275768),
        (0.569832064287, 0.705414478982), (0.635195811927, 0.71565572516),
        (0.69951672331, 0.739419187253), (0.639447159575, 0.805236879972),
        (0.576410514055, 0.835436670169), (0.525398405766, 0.841706377792),
        (0.47641545769, 0.837505914975), (0.41379548902, 0.810045601727),
        (0.380084785646, 0.749979603086), (0.477955996282, 0.74513234612),
        (0.523389793327, 0.748924302636), (0.571057789237, 0.74332894691),
        (0.672409137852, 0.744177032192), (0.572539621444, 0.776609286626),
        (0.5240106503, 0.783370783245), (0.477561227414, 0.778476346951)])

    TPL_MIN, TPL_MAX = np.min(TEMPLATE, axis=0), np.max(TEMPLATE, axis=0)
    MINMAX_TEMPLATE = (TEMPLATE - TPL_MIN) / (TPL_MAX - TPL_MIN)
    print TPL_MIN
    print TPL_MAX
    # print MINMAX_TEMPLATE


def sample_weighting():

    # 2 input samples, 3 reference class samples
    dist = np.array([[1,2,3], [2,3,4], [1,2,3], [2,3,4]])
    weights = np.array([0.2,0.5,0.8])
    print np.shape(dist)
    print np.shape(weights)
    out = dist * weights
    print dist
    print out


def pairwise_subtract():

    # dim = 2
    # ref cluster = 4pts
    ref = np.array([[1,2], [3.4,5], [9,0], [2,4]])

    # samples: 2
    samples = np.array([[2,1],[4,2]])


pairwise_subtract()