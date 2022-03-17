import time
from collections import defaultdict

_TICTOCDICT = defaultdict(float)
_SECDICT = defaultdict(list)


class TicToc(object):
    """
    A simple code timer.
 
    Example
    -------
    >>> with TicToc():
    ...     time.sleep(2)
    Elapsed time is 2.000073 seconds.
    """
    def __init__(self, active=True, do_print=True, key='', accumulate=True, clear=False, sec_key=None):
        if clear:
            self.clear(key)
        self.active = active
        self.do_print = do_print
        self.key = key
        self.accumulate = accumulate
        self.sec_key = sec_key

    def __enter__(self):
        if self.active:
            self.start_time = time.time()
        return self

    def __exit__(self, _type, value, traceback):
        if self.active:
            dt = time.time() - self.start_time
            if self.accumulate:
                _TICTOCDICT[self.key] += dt
            else:
                _TICTOCDICT[self.key] = dt
            if self.do_print:
                overall = " (overall %f)" % _TICTOCDICT[self.key] if self.accumulate else ""
                print("{}{} Elapsed time is {} seconds {}.".format(self.sec_key + "-" if self.sec_key is not None else
                                                                   "", self.key, dt, overall))
            if self.sec_key is not None:
                if self.key not in _SECDICT[self.sec_key]:
                    _SECDICT[self.sec_key].append(self.key)
            else:
                if self.key not in _SECDICT["no-sec"]:
                    _SECDICT["no-sec"].append(self.key)

    @staticmethod
    def get(key):
        return _TICTOCDICT[key]

    @staticmethod
    def keys():
        return _TICTOCDICT.keys()

    @staticmethod
    def items():
        return _TICTOCDICT.items()

    @staticmethod
    def clear(key=None):
        if key is None:
            _TICTOCDICT.clear()
            _SECDICT.clear()
        else:
            _TICTOCDICT.pop(key)

    @staticmethod
    def sortedTimes(time_sorted=True, sec_sorted=False):
        """
        Prints a list of
        :return:
        """
        import operator

        width = max([len(key) for key in TicToc.keys()]) + 3
        fwidth = width + 20 + 2

        frame = ["=" for _ in range(fwidth)]
        frame = ["o"] + frame + ["o"]

        print(''.join(frame))

        import math
        header = "Sorted tasks by amount of time"
        design1 = [' ' for _ in range(int(math.floor(0.5*(fwidth - len(header)))))]
        design2 = [' ' for _ in range(int(math.ceil(0.5*(fwidth - len(header)))))]
        header = "|" + ''.join(design1) + header + ''.join(design2) + "|"

        print(header)
        print(''.join(frame))
        if sec_sorted is True:
            for key, value_list in _SECDICT.items():
                print("| {: <{}} ".format(key, width) + "                    |")
                sub_dict = defaultdict(float)
                for value in value_list:
                    sub_dict[value] = _TICTOCDICT[value]
                if time_sorted is True:
                    for data in sorted(sub_dict.items(), key=operator.itemgetter(1), reverse=True):
                        print("|   " + '{: <{}}'.format(data[0], width) + " %f " % (data[1]) + " seconds |")
                else:
                    for data in sub_dict.items():
                        print("|   " + '{: <{}}'.format(data[0], width) + " %f " % (data[1]) + " seconds |")
        else:
            if time_sorted is True:
                for data in sorted(_TICTOCDICT.items(), key=operator.itemgetter(1), reverse=True):
                    print("| " + '{: <{}}'.format(data[0], width) + " %f " % (data[1]) + " seconds |")
        print(''.join(frame))
