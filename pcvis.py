import sys
import os
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata

def test():
    print("Hello World!")

def show3dgraph(px,py,pz,grid_step=100):

    x = np.array(px)
    y = np.array(py)
    z = np.array(pz)

    # x = np.array([25, 25, 25, 50, 50, 50, 75, 75, 75, 100, 100, 100])
    # y = np.array([2500, 5000, 7500, 2500, 5000, 7500, 2500, 5000, 7500, 2500, 5000, 7500])
    # z = np.array([0.944, 0.4719, 0.3145, 0.9279, 0.4699, 0.3083, 0.2344, 0.1168, 0.0778, 0.9165, 0.4627, 0.3093])

    # создаем новую сетку для интерполяции
    xi = np.linspace(min(x), max(x), grid_step)
    yi = np.linspace(min(y), max(y), grid_step)
    Xi, Yi = np.meshgrid(xi, yi)

    # интерполируем значения функции на новой сетке
    Zi = griddata((x, y), z, (Xi, Yi), method='linear')

    # строим поверхность
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(Xi, Yi, Zi, cmap='hot')

    # настраиваем отображение
    ax.set_xlabel


def visualize_rotate(data):
    x_eye, y_eye, z_eye = 1.25, 1.25, 0.8
    frames = []

    def rotate_z(x, y, z, theta):
        w = x + 1j * y
        return np.real(np.exp(1j * theta) * w), np.imag(np.exp(1j * theta) * w), z

    for t in np.arange(0, 10.26, 0.1):
        xe, ye, ze = rotate_z(x_eye, y_eye, z_eye, -t)
        frames.append(dict(layout=dict(scene=dict(camera=dict(eye=dict(x=xe, y=ye, z=ze))))))
    fig = go.Figure(data=data,
                    layout=go.Layout(
                        updatemenus=[dict(type='buttons',
                                          showactive=False,
                                          y=1,
                                          x=0.8,
                                          xanchor='left',
                                          yanchor='bottom',
                                          pad=dict(t=45, r=10),
                                          buttons=[dict(label='Play',
                                                        method='animate',
                                                        args=[None, dict(frame=dict(duration=50, redraw=True),
                                                                         transition=dict(duration=0),
                                                                         fromcurrent=True,
                                                                         mode='immediate'
                                                                         )]
                                                        )
                                                   ]
                                          )
                                     ]
                    ),
                    frames=frames
                    )

    return fig


def pcshow(xs, ys, zs):
    data = [go.Scatter3d(x=xs, y=ys, z=zs,
                         mode='markers')]
    fig = visualize_rotate(data)
    fig.update_traces(marker=dict(size=2,
                                  line=dict(width=2,
                                            color='DarkSlateGrey')),
                      selector=dict(mode='markers'))
    fig.show()


def pcshow_xyz(data, x_column=0, y_column=1, z_column=2, frac=1):
    xs=data[:,x_column]
    ys=data[:,y_column]
    zs=data[:,z_column]
    idx = np.random.randint(len(xs), size=int(len(xs) * frac))
    xs=xs[idx]
    ys=ys[idx]
    zs=zs[idx]
    pcshow(xs,ys,zs)


def pcshow_xyzl(data, x_column=0, y_column=1, z_column=2, label_column=6, frac=1):
    cdict = {
        0: 'red',
        1: 'orange',
        2: 'green',
        3: 'blue',
        4: 'purple',
        5: 'black',
        6: 'yellow',
    }
    xs = data[:, x_column]
    ys = data[:, y_column]
    zs = data[:, z_column]
    labels = np.int64(data[:, label_column])
    idx = np.random.randint(len(xs), size=int(len(xs) * frac))
    xs=xs[idx]
    ys=ys[idx]
    zs=zs[idx]
    labels=labels[idx]
    data = [go.Scatter3d(x=xs, y=ys, z=zs, mode='markers')]
    fig = visualize_rotate(data)
    fig.update_traces(marker=dict(
                        size=2,
                        color=[cdict[label] for label in labels]),
                      selector=dict(mode='markers'))
    fig.show()


def pcshow_xyzrgb(data, x_column=0, y_column=1, z_column=2, r_column=3, g_column=4, b_column=5, frac=0.5):
    xs = data[:, x_column]
    ys = data[:, y_column]
    zs = data[:, z_column]
    colors = np.int64(data[:, [r_column,g_column,b_column]])
    idx = np.random.randint(len(xs), size=int(len(xs) * frac))
    xs = xs[idx]
    ys = ys[idx]
    zs = zs[idx]
    colors = colors[idx]
    data = [go.Scatter3d(x=xs, y=ys, z=zs, mode='markers')]
    fig = visualize_rotate(data)
    fig.update_traces(marker=dict(
                        size=2,
                        color=colors),
                      selector=dict(mode='markers'))
    fig.show()


def show_result_graph(path_file):
    if not os.path.isfile(path_file):
        sys.exit("file not found")

    data=np.loadtxt(path_file,delimiter=" ")
    currdir=os.path.dirname(path_file)
    plt.subplot(211)
    plt.plot(data[:,[0]])
    plt.subplot(212)
    plt.plot(data[:,[1]])
    plt.savefig(os.path.join(currdir,"gr.png"))
