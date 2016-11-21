#!/usr/bin/python
import numpy as np
import struct
import sys


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


test_type_conversion()