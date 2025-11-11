import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # necessary for 3D plotting

def run_task3():
    # ---------------------------------------------------------
    # Torus parameters
    # ---------------------------------------------------------
    R = 5  # distance from center of tube to center of torus
    r = 2  # radius of tube

    # Create meshgrid for u, v
    u = np.linspace(0, 2 * np.pi, 60)
    v = np.linspace(0, 2 * np.pi, 30)
    u, v = np.meshgrid(u, v)

    # Parametric equations of the torus
    def torus(R, r, u, v):
        x = (R + r * np.cos(v)) * np.cos(u)
        y = (R + r * np.cos(v)) * np.sin(u)
        z = r * np.sin(v)
        return x, y, z

    X, Y, Z = torus(R, r, u, v)

    # ---------------------------------------------------------
    # Create 3D figure
    # ---------------------------------------------------------
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-R-r-1, R+r+1)
    ax.set_ylim(-R-r-1, R+r+1)
    ax.set_zlim(-r-1, r+1)
    ax.set_box_aspect([1,1,1])

    # Initial surface plot
    surface = [ax.plot_surface(X, Y, Z, color='cyan', edgecolor='k')]

    # ---------------------------------------------------------
    # Animation function
    # ---------------------------------------------------------
    def update(frame):
        ax.view_init(elev=30, azim=frame)  # rotate around z-axis
        return surface

    # ---------------------------------------------------------
    # Create animation
    # ---------------------------------------------------------
    frames = np.linspace(0, 360, 200)  # 200 frames for smooth rotation
    anim = FuncAnimation(fig, update, frames=frames, interval=50, blit=False)

    # ---------------------------------------------------------
    # Save as GIF
    # ---------------------------------------------------------
    anim.save('rotating_torus.gif', writer='pillow', fps=20)

    plt.show()
