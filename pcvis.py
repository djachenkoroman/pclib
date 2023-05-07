import os
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt

def test():
    print("Hello World!")


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
    data=np.loadtxt(path_file,delimiter=" ")
    currdir=os.path.basename(os.path.dirname(path_file))
    print(currdir)
