import os


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def get_print(path):
    def func(*args):
        with open(os.path.join(path, 'log.txt'), 'a') as f:
            print(*args, file=f)
        print(*args)
    return func
