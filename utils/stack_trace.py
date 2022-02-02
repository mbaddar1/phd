import sys
import traceback


def g():
    traceback.print_stack(file=sys.stdout)
    pass


def f():
    g()

if __name__ == '__main__':
    f()