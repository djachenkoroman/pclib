import os
import numpy as np
import plotly.graph_objects as go
import plotly.express as px


def preprocess(data_root, data_dir, GRID_SIZE):
    os.makedirs(data_dir, exist_ok=False)
    data = np.loadtxt(data_root)
    classes = set(data[:, -1])
    num_classes = len(classes)
    idx = 0
    x = data[:, 0]
    y = data[:, 1]
    x_max = int(np.max(x)) + 1
    x_min = int(np.min(x)) - 1
    y_max = int(np.max(y)) + 1
    y_min = int(np.min(y)) - 1
    del x
    del y

    for i in range(x_min, x_max - GRID_SIZE, GRID_SIZE):
        for j in range(y_min, y_max - GRID_SIZE, GRID_SIZE):
            arr = data[
                (data[:, 0] > i) & (data[:, 0] < i + GRID_SIZE) & (data[:, 1] > j) & (data[:, 1] < j + GRID_SIZE)]
            np.save(f'{data_dir}/{idx}.npy', arr)
            idx += 1
    del data
    return num_classes, classes
