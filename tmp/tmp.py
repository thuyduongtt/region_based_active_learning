import numpy as np
import tensorflow as tf


def test_print(*content, file):
    print(*content)
    if file is not None:
        print(*content, file=file)


if __name__ == '__main__':
    print(__file__)
    log_file = open('log.txt', 'a')
    log_file = open('log.txt', 'a')
    log_file = open('log.txt', 'a')
    test_print(10, 'is', 'bigger than', 9, 'ok?', file=log_file)
    log_file.close()
