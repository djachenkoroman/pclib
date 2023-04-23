import os
import pandas as pd
import torch
import torch.nn.parallel
import torch.utils.data
import numpy as np
from pointnet import PointNetDenseCls


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


def preprocess2(data_root, data_dir, grid_size):
    dbg=True
    fmt = '%1.6f', '%1.6f', '%1.6f', '%d', '%d', '%d', '%d'
    os.makedirs(data_dir, exist_ok=False)
    data = np.loadtxt(data_root)
    classes = set(data[:, -1])
    num_classes = len(classes)
    if data.shape[1]==4:
        fmt = '%1.6f', '%1.6f', '%1.6f', '%d'
    elif data.shape[1]==7:
        fmt = '%1.6f', '%1.6f', '%1.6f', '%d', '%d', '%d', '%d'
    if dbg: print("params\ndata_root:{0}\ndata_dir:{1}\nGRID_SIZE:{2}\ndata.shape:{3}\nclasses:{4}\nnum_classes:{5}\nfmt:{6}".format(data_root, data_dir, grid_size, data.shape,classes),num_classes,fmt)

    idx = 0
    x = data[:, 0]
    y = data[:, 1]
    x_max = int(np.max(x)) + 1
    x_min = int(np.min(x)) - 1
    y_max = int(np.max(y)) + 1
    y_min = int(np.min(y)) - 1
    del x
    del y

    for i in range(x_min, x_max - grid_size, grid_size):
        for j in range(y_min, y_max - grid_size, grid_size):
            arr = data[
                (data[:, 0] > i) & (data[:, 0] < i + grid_size) & (data[:, 1] > j) & (data[:, 1] < j + grid_size)]
            fn='{0}/{1}.txt'.format(data_dir,str(idx).zfill(5))
            if dbg: print(fn)
            np.savetxt(fn,arr,delimiter=',',fmt=fmt)
            idx += 1
    del data
    return num_classes, classes


def pointcloud_pointnet_seg(
        data,
        pred_path,
        checkpoint_path,
        channels=6,
        num_classes=5,
        GRID_SIZE=50,
        device='cpu',
        npoints=2400
):
    model = PointNetDenseCls(channels=channels, num_classes=num_classes)
    # model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')))
    model.load_state_dict(torch.load(checkpoint_path))
    model.to(device)
    model.eval()
    idx = 0
    x = data[:, 0]
    y = data[:, 1]
    x_max = int(np.max(x)) + 1
    x_min = int(np.min(x)) - 1
    y_max = int(np.max(y)) + 1
    y_min = int(np.min(y)) - 1
    del x
    del y
    preds = []
    points = []

    for i in range(x_min, x_max - GRID_SIZE, GRID_SIZE):
        for j in range(y_min, y_max - GRID_SIZE, GRID_SIZE):
            choice = np.random.choice(len(data), npoints, replace=True)
            cdata = data[choice, :]
            if cdata.shape[1] == 3:
                coord = cdata
            else:
                coord, rgb = cdata[:, :3], cdata[:, 3:]
            coord = coord - np.expand_dims(np.mean(coord, axis=0), 0)  # center
            dist = np.max(np.sqrt(np.sum(coord ** 2, axis=1)), 0)
            coord = coord / dist  # scale
            if cdata.shape[1] == 6:
                arr = np.hstack([coord, rgb]).astype(np.float32)
            else:
                arr = coord.astype(np.float32)
            points.append(arr)
            arr = torch.FloatTensor(arr).to(device)
            arr = torch.unsqueeze(arr, dim=0)
            arr = arr.transpose(2, 1)
            pred_tuple = model(arr)
            if type(pred_tuple) == tuple:
                if len(pred_tuple) == 2:
                    pred, _ = pred_tuple
                elif len(pred_tuple) == 3:
                    pred, _, _ = pred_tuple
            else:
                pred = pred_tuple

            pred_choice = pred.data.max(2)[1]
            pred = pred_choice.cpu().data.numpy()
            preds.append(pred)

            del arr

    del model
    output = []
    for point, label in zip(points, preds):
        label = label.reshape(npoints, 1)
        output.append(np.hstack([point, label]))

    output = np.vstack(output)
    df = pd.DataFrame(output)
    n = df.shape[1] - 1
    df[n] = df[n].astype(int)
    df.to_csv(pred_path, index=False)
    return preds, points


def pointcloud_pointnet_seg_real(
        data,
        pred_path,
        checkpoint_path,
        channels=6,
        num_classes=5,
        GRID_SIZE=50,
        device='cpu',
        npoints=2400
):
    print("channels={0}\nnum_classes={1}\nnpoints={2}".format(channels,num_classes,npoints))
    model = PointNetDenseCls(channels=channels, num_classes=num_classes)
    # model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')))
    model.load_state_dict(torch.load(checkpoint_path))
    model.to(device)
    model.eval()
    idx = 0
    x = data[:, 0]
    y = data[:, 1]
    x_max = int(np.max(x)) + 1
    x_min = int(np.min(x)) - 1
    y_max = int(np.max(y)) + 1
    y_min = int(np.min(y)) - 1
    del x
    del y
    preds = []
    points = []
    points_in = []

    for i in range(x_min, x_max - GRID_SIZE, GRID_SIZE):
        for j in range(y_min, y_max - GRID_SIZE, GRID_SIZE):
            choice = np.random.choice(len(data), npoints, replace=True)
            cdata = data[choice, :]
            if channels == 3:
                coord = cdata[:, :3]
            else:
                coord, rgb = cdata[:, :3], cdata[:, 3:]
            coord = coord - np.expand_dims(np.mean(coord, axis=0), 0)  # center
            dist = np.max(np.sqrt(np.sum(coord ** 2, axis=1)), 0)
            coord = coord / dist  # scale
            if channels == 6:
                arr = np.hstack([coord, rgb]).astype(np.float32)
            else:
                arr = coord.astype(np.float32)
            points.append(arr)
            points_in.append(cdata[:, :3])
            arr = torch.FloatTensor(arr).to(device)
            arr = torch.unsqueeze(arr, dim=0)
            arr = arr.transpose(2, 1)
            pred_tuple = model(arr)
            print(len(pred_tuple))
            if type(pred_tuple) == tuple:
                if len(pred_tuple) == 2:
                    pred, _ = pred_tuple
                elif len(pred_tuple) == 3:
                    pred, _, _ = pred_tuple
            else:
                pred = pred_tuple
            pred_choice = pred.data.max(2)[1]
            pred = pred_choice.cpu().data.numpy()
            preds.append(pred)

            del arr

    del model
    output = []
    for point, label in zip(points_in, preds):
        label = label.reshape(npoints, 1)
        output.append(np.hstack([point, label]))

    output = np.vstack(output)
    df = pd.DataFrame(output)
    n = df.shape[1] - 1
    df[n] = df[n].astype(int)
    df.to_csv(pred_path, index=False)
    return preds, points_in


def pointcloud_pointnet_seg_real2(
        data,
        pred_path,
        checkpoint_path,
        channels=6,
        num_classes=5,
        GRID_SIZE=50,
        device='cpu',
        npoints=2400
):
    debug=True
    if debug: print("model:{0}\nchannels:{1}\nnum_classes:{2}\ndata.shape:{3}".format(checkpoint_path,channels,num_classes,data.shape))
    model = PointNetDenseCls(channels=channels, num_classes=num_classes)
    model.load_state_dict(torch.load(checkpoint_path))
    model.to(device)
    model.eval()
    idx = 0
    x = data[:, 0]
    y = data[:, 1]
    x_max = int(np.max(x)) + 1
    x_min = int(np.min(x)) - 1
    y_max = int(np.max(y)) + 1
    y_min = int(np.min(y)) - 1
    del x
    del y
    preds = []
    points = []

    for i in range(x_min, x_max - GRID_SIZE, GRID_SIZE):
        for j in range(y_min, y_max - GRID_SIZE, GRID_SIZE):
            choice = np.random.choice(len(data), npoints, replace=True)
            cdata = data[choice, :]
            print("cdata.shape={0}".format(cdata.shape))
            if cdata.shape[1] == 3:
                coord = cdata
            else:
                coord, rgb = cdata[:, :3], cdata[:, 3:]
            coord = coord - np.expand_dims(np.mean(coord, axis=0), 0)  # center
            dist = np.max(np.sqrt(np.sum(coord ** 2, axis=1)), 0)
            coord = coord / dist  # scale
            if cdata.shape[1] == 6:
                arr = np.hstack([coord, rgb]).astype(np.float32)
            else:
                arr = coord.astype(np.float32)
            points.append(arr)
            arr = torch.FloatTensor(arr).to(device)
            arr = torch.unsqueeze(arr, dim=0)
            arr = arr.transpose(2, 1)
            pred_tuple = model(arr)
            if type(pred_tuple) == tuple:
                if len(pred_tuple) == 2:
                    pred, _ = pred_tuple
                elif len(pred_tuple) == 3:
                    pred, _, _ = pred_tuple
            else:
                pred = pred_tuple

            pred_choice = pred.data.max(2)[1]
            pred = pred_choice.cpu().data.numpy()
            print(pred)
            preds.append(pred)

            del arr

    del model
    output = []
    for point, label in zip(points, preds):
        label = label.reshape(npoints, 1)
        output.append(np.hstack([point, label]))

    output = np.vstack(output)
    df = pd.DataFrame(output)
    n = df.shape[1] - 1
    df[n] = df[n].astype(int)
    df.to_csv(pred_path, index=False)
    return preds, points
