import random
import sys
import math
from uids.utils.DataAnalysis import *
from uids.utils.Logger import Logger as log
from numpy import (array, dot, arccos, clip)
from numpy.linalg import norm
from sklearn.metrics import pairwise_distances
from sklearn.utils.extmath import fast_dot


class SetMeanDistCosine:
    @staticmethod
    def get_score(test_samples, reference_set):
        class_mean = np.mean(reference_set, axis=0)
        return pairwise_distances(class_mean.reshape(1, -1), test_samples, metric='cosine')[0]


class SetMeanDistL2Quared:
    @staticmethod
    def get_score(test_samples, reference_set):
        class_mean = np.mean(reference_set, axis=0)
        return np.square(pairwise_distances(class_mean.reshape(1, -1), test_samples, metric='euclidean')[0])


class ABOD:

    @staticmethod
    def get_set_score(test_samples):

        assert test_samples.ndim == 2

        dist_lookup = pairwise_distances(test_samples, test_samples, metric='euclidean')

        # print np.shape(dist_lookup[0])
        factors = []

        # if only one sample: cannot calculate abof
        if len(test_samples) < 4:
            log.severe('Cannot calculate ABOF with {} reference samples (variance calculation needs at least 4 reference points)'.format(len(test_samples)))
            raise Exception

        for i_sample, A in enumerate(test_samples):
            varList = []
            for i in range(len(test_samples)):

                if i_sample == i:  # ensure A != B
                        continue
                # select first point in reference set
                B = test_samples[i]
                # distance
                AB = dist_lookup[i_sample][i]
                j = 0
                for j in range(i + 1):
                    if j == i or j==i_sample:  # ensure B != C
                        continue
                    # select second point in reference set
                    C = test_samples[j]
                    # distance
                    AC = dist_lookup[i_sample][j]

                    if np.array_equal(B, C):
                        log.error("Points are equal: B == C! Assuming classification of training point (ABOD 1000)")
                        varList.append(1000)
                        print "Bi/Cj: {}/{}".format(i, j)
                        # sys.exit('ERROR\tangleBAC\tmath domain ERROR, |cos<AB, AC>| <= 1')
                        continue

                    angle_BAC = ABOD.angleBAC(A, B, C, AB, AC)
                    # angle_BAC = ABOD.angleFast(A-B, A-C)

                    # compute each element of variance list
                    try:
                        # apply weighting
                        tmp = angle_BAC / float(math.pow(AB * AC, 2))
                    except ZeroDivisionError:
                        log.severe("ERROR\tABOF\tfloat division by zero! Trying to predict training point?'")
                        tmp = 500
                        # sys.exit('ERROR\tABOF\tfloat division by zero! Trying to predict training point?')
                    varList.append(tmp)
            factors.append(np.var(varList))
        return np.array(factors)

    @staticmethod
    def get_score(test_samples, reference_set):

        assert test_samples.ndim == 2
        assert reference_set.ndim == 2

        dist_lookup = pairwise_distances(test_samples, reference_set, metric='euclidean')

        # print np.shape(dist_lookup[0])
        factors = []

        # if only one sample: cannot calculate abof
        if len(reference_set) < 3:
            log.severe('Cannot calculate ABOF with {} reference samples (variance calculation needs at least 3 reference points)'.format(len(reference_set)))
            raise Exception

        for i_sample, A in enumerate(test_samples):
            varList = []
            for i in range(len(reference_set)):
                # select first point in reference set
                B = reference_set[i]
                # distance
                AB = dist_lookup[i_sample][i]
                j = 0
                for j in range(i + 1):
                    if j == i:  # ensure B != C
                        continue
                    # select second point in reference set
                    C = reference_set[j]
                    # distance
                    AC = dist_lookup[i_sample][j]

                    if np.array_equal(B, C):
                        log.error("Points are equal: B == C! Assuming classification of training point (ABOD 1000)")
                        varList.append(1000)
                        print "Bi/Cj: {}/{}".format(i, j)
                        # sys.exit('ERROR\tangleBAC\tmath domain ERROR, |cos<AB, AC>| <= 1')
                        continue

                    angle_BAC = ABOD.angleBAC(A, B, C, AB, AC)
                    # angle_BAC = ABOD.angleFast(A-B, A-C)

                    # compute each element of variance list
                    try:
                        # apply weighting
                        tmp = angle_BAC / float(math.pow(AB * AC, 2))
                    except ZeroDivisionError:
                        log.severe("ERROR\tABOF\tfloat division by zero! Trying to predict training point?'")
                        tmp = 500
                        # sys.exit('ERROR\tABOF\tfloat division by zero! Trying to predict training point?')
                    varList.append(tmp)
            factors.append(np.var(varList))
        return np.array(factors)

    @staticmethod
    def angleSlow(u, v):
        c = np.dot(u, v) / norm(u) / norm(v)  # -> cosine of the angle
        angle = arccos(clip(c, -1, 1))  # if you really want the angle
        return angle

    @staticmethod
    def angleBAC(A, B, C, AB, AC):				# AB AC mold
        """
        calculate: <AB, AC> = |AB||AC|*cos(AB,AC)
        """
        vector_AB = B - A						# vector_AB = (x1, x2, ..., xn)
        vector_AC = C - A
        # fast np implementation
        dotProduct = np.dot(vector_AB, vector_AC)
        # dotProduct = fast_dot(vector_AB, vector_AC)

        try:
            cos_AB_AC_ = dotProduct / (AB * AC)  # cos<AB, AC>
        except ZeroDivisionError:
            sys.exit('ERROR\tangleBAC\tdistance can not be zero!')

        # alternative: clip(cos_AB_AC_, -1, 1)

        cos_AB_AC_ = clip(cos_AB_AC_, -1, 1)

        if math.fabs(cos_AB_AC_) > 1:

            print "cos<AB, AC> = ", cos_AB_AC_
            if np.array_equal(B, C):
                log.error("Points are equal: B == C")
            elif np.array_equal(A, B):
                log.error("Points are equal: A == B")
            elif np.array_equal(A, C):
                log.error("Points are equal: A == C")

            print 'AB = %f, AC = %f' % (AB, AC)
            print 'AB * AC = ', dotProduct
            print '|AB| * |AC| = ', AB * AC
            sys.exit('ERROR\tmath domain ERROR, |cos<AB, AC>| <= 1')

        # faster than np.arccos
        angle = float(math.acos(cos_AB_AC_))    # <AB, AC> = arccos(cos<AB, AC>)
        return angle
