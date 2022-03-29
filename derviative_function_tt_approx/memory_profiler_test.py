import numpy as np
from memory_profiler import profile, memory_usage


@profile
def f1():
    size = int(1e8)
    l1 = list(np.zeros(size))
    l2 = list(np.zeros(size))
    l1.extend(l2)
    """
Filename: /home/mbaddar/Documents/mbaddar/phd/phd/derviative_function_tt_approx/memory_profiler_test.py

Line #    Mem usage    Increment  Occurrences   Line Contents
=============================================================
     5     33.0 MiB     33.0 MiB           1   @profile
     6                                         def f1():
     7     33.0 MiB      0.0 MiB           1       size = int(1e8)
     8   3897.3 MiB   3864.3 MiB           1       l1 = list(np.zeros(size))
     9   7761.1 MiB   3863.9 MiB           1       l2 = list(np.zeros(size))
    10   8524.0 MiB    762.9 MiB           1       l1.extend(l2)



Process finished with exit code 0

    """


@profile
def f2(size):
    l2 = []
    for i in range(10):
        l2.extend(list(np.zeros(size)))
        print(len(l2))


@profile
def f3(l):
    l.append(1)


if __name__ == '__main__':
    # f1()
    # N = 8
    # mem_usage_iter = []
    # for i in range(N):
    #     size = int(np.power(10, i))
    #     proc = (f2, (size,), {})
    #     m = memory_usage(proc=proc, max_usage=True, retval=True)
    #     mem_usage_iter.append(m[0])
    # print(mem_usage_iter)
    # print('finished')
    l = []
    proc = (f3, (l,), {})

    for i in range(6):
        m = memory_usage(proc=proc, max_usage=True, retval=False, max_iterations=1)
        # f3(l)
        print(len(l))
