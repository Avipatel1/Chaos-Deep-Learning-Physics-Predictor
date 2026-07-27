import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

if not os.path.exists("chaos_training_data.txt"):
    print("Error: chaos_training_data.txt missing. Execute C++ generator first.")
    exit()

# 1. Ingest raw matrix (Columns: theta1, theta2, omega1, omega2)
data = np.loadtxt("chaos_training_data.txt")
total_steps = len(data)

# Fixed mechanical rod dimensions (must match C++ file parameters)
L1, L2 = 1.0, 1.0

# 2. Convert raw angles to 2D Cartesian spatial coordinates using trig
t1 = data[:, 0]
t2 = data[:, 1]

x1 = L1 * np.sin(t1)
y1 = -L1 * np.cos(t1)

x2 = x1 + L2 * np.sin(t2)
y2 = y1 - L2 * np.cos(t2)

# 3. Configure deep space plotting canvas
fig, ax = plt.subplots(figsize=(7, 7), facecolor='#0B0C10')
ax.set_facecolor('#0B0C10')
ax.set_xlim(-2.2, 2.2)
ax.set_ylim(-2.2, 2.2)
ax.set_aspect('equal')
ax.axis('off')

# Graphic elements: The pendulum arms, the mass points, and the path trail
line, = ax.plot([], [], 'o-', color='#457B9D', linewidth=3, markersize=10, markerfacecolor='#E05A47')
trail, = ax.plot([], [], '-', color='#1F2833', alpha=0.6, linewidth=1.5)

# Array tracking past trajectory history
trail_x, trail_y = [], []

# 4. Canvas Clock Animation Loop
def update(frame):
    global trail_x, trail_y
    
    # Speed up animation playback by stepping through blocks of data
    step_idx = frame * 4
    if step_idx >= total_steps:
        step_idx = total_steps - 1

    # Update structural link lines: Origin (0,0) -> Joint 1 -> Mass 2
    this_x = [0, x1[step_idx], x2[step_idx]]
    this_y = [0, y1[step_idx], y2[step_idx]]
    line.set_data(this_x, this_y)

    # Append coordinate states to append visual trail history
    trail_x.append(x2[step_idx])
    trail_y.append(y2[step_idx])
    
    # Cap the history trail to prevent canvas lag
    if len(trail_x) > 300:
        trail_x.pop(0)
        trail_y.pop(0)
        
    trail.set_data(trail_x, trail_y)
    return line, trail

ani = animation.FuncAnimation(fig, update, frames=total_steps // 4, interval=15, blit=True, repeat=True)
plt.title("Visual Analytics: Double Pendulum Chaotic Trajectory", color='white', fontsize=12, pad=10)
plt.show()
