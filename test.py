import numpy as np

import py_vec_add

a = np.arange(257, dtype=np.int32)
b = np.arange(257, dtype=np.int32)

c = py_vec_add.vec_add_naive_int(a, b)
