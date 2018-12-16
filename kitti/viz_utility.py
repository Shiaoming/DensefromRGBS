import numpy as np

vis_ok = False
try:
    import mayavi.mlab

    vis_ok = True
except:
    print("mayavi.mlab not available")


# def pc_viewer(points, point_size=0.02, seg=None, color_map=None):
def pc_viewer(points, seg=None, color_map=None, figure=None, show=True):
    if vis_ok:
        x = points[:, 0]  # x position of point
        y = points[:, 1]  # y position of point
        z = points[:, 2]  # z position of point

        N = x.shape[0]
        scalars = np.arange(N)

        if seg is None:
            if figure != None:
                mayavi.mlab.points3d(x, y, z, scalars, mode="sphere", figure=figure)
            else:
                mayavi.mlab.points3d(x, y, z, scalars, mode="sphere")
        else:
            # construct color of each point
            color = np.random.random((N, 4)).astype(np.uint8)
            color[:, -1] = 255  # No transparency
            for i, color_idx in enumerate(seg):
                color[i, 0:3] = 255 * np.array(color_map[color_idx])  # assign color

            if figure != None:
                nodes = mayavi.mlab.points3d(x, y, z, scalars, mode="sphere", figure=figure)
            else:
                nodes = mayavi.mlab.points3d(x, y, z, scalars, mode="sphere")

            nodes.glyph.scale_mode = 'data_scaling_off'
            nodes.glyph.color_mode = 'color_by_scalar'
            # Set look-up table and redraw
            nodes.module_manager.scalar_lut_manager.lut.table = color

        mayavi.mlab.orientation_axes()
        # mayavi.mlab.xlabel('x')
        # mayavi.mlab.ylabel('y')
        # mayavi.mlab.zlabel('z')

        if show:
            mayavi.mlab.show()
    else:
        print("mayavi.mlab not available")

def draw_lidar_simple(pc, color=None):
    ''' Draw lidar points. simplest set up. '''
    fig = mayavi.mlab.figure(figure=None, bgcolor=(0,0,0), fgcolor=None, engine=None, size=(1600, 1000))
    if color is None: color = pc[:,2]
    #draw points
    mayavi.mlab.points3d(pc[:,0], pc[:,1], pc[:,2], color, color=None, mode='point', colormap = 'gnuplot', scale_factor=1, figure=fig)
    #draw origin
    mayavi.mlab.points3d(0, 0, 0, color=(1,1,1), mode='sphere', scale_factor=0.2)
    #draw axis
    # axes=np.array([
    #     [2.,0.,0.,0.],
    #     [0.,2.,0.,0.],
    #     [0.,0.,2.,0.],
    # ],dtype=np.float64)

    mayavi.mlab.orientation_axes()

    # mayavi.mlab.xlabel('x')
    # mayavi.mlab.ylabel('y')
    # mayavi.mlab.zlabel('x')

    # mayavi.mlab.plot3d([0, axes[0,0]], [0, axes[0,1]], [0, axes[0,2]], color=(1,0,0), tube_radius=None, figure=fig)
    # mayavi.mlab.plot3d([0, axes[1,0]], [0, axes[1,1]], [0, axes[1,2]], color=(0,1,0), tube_radius=None, figure=fig)
    # mayavi.mlab.plot3d([0, axes[2,0]], [0, axes[2,1]], [0, axes[2,2]], color=(0,0,1), tube_radius=None, figure=fig)
    mayavi.mlab.view(azimuth=180, elevation=70, focalpoint=[ 12.0909996 , -1.04700089, -2.03249991], distance=62.0, figure=fig)
    return fig