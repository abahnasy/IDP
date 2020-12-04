'''
Visualize a a second by passinig the tfrecord of the sequence as argument and then output a 2d projected view of the frames
option to save the images in _out folder

'''
axes_limits = [
    [-80, 80], # X axis range
    [-80, 80], # Y axis range
    [-3, 10]   # Z axis range
]
axes_str = ['X', 'Y', 'Z']

colors = {
    0: 'b',
    1: 'r',
    2: 'g',
    3: 'c',
    4: 'm'
}


def draw_box(pyplot_axis, vertices, axes=[0, 1, 2], color='black'):
    """
    Draws a bounding 3D box in a pyplot axis.
    
    Parameters
    ----------
    pyplot_axis : Pyplot axis to draw in.
    vertices    : Array 8 box vertices containing x, y, z coordinates.
    axes        : Axes to use. Defaults to `[0, 1, 2]`, e.g. x, y and z axes.
    color       : Drawing color. Defaults to `black`.
    """
    vertices = vertices[axes, :]
    connections = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # Lower plane parallel to Z=0 plane
        [4, 5], [5, 6], [6, 7], [7, 4],  # Upper plane parallel to Z=0 plane
        [0, 4], [1, 5], [2, 6], [3, 7]  # Connections between upper and lower planes
    ]
    for connection in connections:
        pyplot_axis.plot(*vertices[:, connection], c=color, lw=0.5)



def draw_point_cloud(ax, title, axes=[0, 1, 2], xlim3d=None, ylim3d=None, zlim3d=None):
        """
        Convenient method for drawing various point cloud projections as a part of frame statistics.
        """
        ax.scatter(*np.transpose(pc[:, axes]),s=0.02, cmap='gray')
        ax.set_title(title)
        ax.set_xlabel('{} axis'.format(axes_str[axes[0]]))
        ax.set_ylabel('{} axis'.format(axes_str[axes[1]]))
        if len(axes) > 2:
            ax.set_xlim3d(*axes_limits[axes[0]])
            ax.set_ylim3d(*axes_limits[axes[1]])
            ax.set_zlim3d(*axes_limits[axes[2]])
            ax.set_zlabel('{} axis'.format(axes_str[axes[2]]))
        else:
            ax.set_xlim(*axes_limits[axes[0]])
            ax.set_ylim(*axes_limits[axes[1]])
        # User specified limits
        if xlim3d!=None:
            ax.set_xlim3d(xlim3d)
        if ylim3d!=None:
            ax.set_ylim3d(ylim3d)
        if zlim3d!=None:
            ax.set_zlim3d(zlim3d)
            
        for i in range(labels.shape[0]):
            label_type = labels[i][0] # get label
#             box_corners = np.transpose(t[1])
#             box_corners = t[1]
#             print(box_corners)
            box_corners = get_corners_from_labels_array(labels[i])
            draw_box(ax, box_corners, axes=axes, color=colors[label_type])
#             break

        

# Draw point cloud data as 3D plot
f2 = plt.figure(figsize=(30, 30))
ax2 = f2.add_subplot(111, projection='3d')                    
draw_point_cloud(ax2, 'Velodyne scan', xlim3d=(-10,30))
plt.show()

# Draw point cloud data as plane projections
f, ax3 = plt.subplots(1, 1, figsize=(25, 25))
# draw_point_cloud(
#     ax3[0], 
#     'Velodyne scan, XZ projection (Y = 0), the car is moving in direction left to right', 
#     axes=[0, 2] # X and Z axes
# )
draw_point_cloud(
    ax3, 
    'Velodyne scan, XY projection (Z = 0), the car is moving in direction left to right', 
    axes=[0, 1] # X and Y axes
)
# draw_point_cloud(
#     ax3[2], 
#     'Velodyne scan, YZ projection (X = 0), the car is moving towards the graph plane', 
#     axes=[1, 2] # Y and Z axes
# )
plt.show()