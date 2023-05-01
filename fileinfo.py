import os
import sys
import numpy as np
from tqdm import tqdm
import argparse
from art import *

parser = argparse.ArgumentParser()
parser.add_argument('--dsfile', type=str, default='', help='dataset file [default=null]')

args = parser.parse_args()

if __name__ == '__main__':
    tprint("fileinfo")
    data = np.loadtxt(args.dsfile, delimiter=',')
    print("columns: {0}\nrows: {1}".format(data.shape[1],data.shape[0]))


    print("Example")
    print("--------------------------------------")

    # получим объект файла
    file1 = open(args.dsfile, "r")

    for i in range(10):
        # считываем строку
        line = file1.readline()
        # прерываем цикл, если строка пустая
        if not line:
            break
        # выводим строку
        print(line.strip())

    # закрываем файл
    file1.close
    print("--------------------------------------")
