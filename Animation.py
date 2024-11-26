import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Define constants
a = 0.2
L = 1.0
N = 51
dx = L / (N - 1)
dt = 0.5 * dx / np.abs(a)
x = np.linspace(0, L, N)
T = 2.5

# Define initial condition
def u0(x):
    return np.sin(2 * np.pi * x)

# Crank-Nicolson Scheme
def crank_nicolson_method(a, T):
    dt = 0.5 * dx / a
    u = u0(x + a * T)  # Initial condition at t = 0
    for t in np.arange(0, T, dt):
        u = crank_nicolson_method_update(u, a, dt, dx)

    return u

# Implement Crank-Nicolson discretization scheme
def crank_nicolson_method_update(u, a, dt, dx):
    lambd = a * dt / (2 * dx)
    N = len(u) - 1

    # Construct the tridiagonal matrix A
    A = np.diagflat([-lambd] * N, -1) + np.diagflat([1 + 2 * lambd] * (N + 1)) + np.diagflat([-lambd] * N, 1)

    # Construct the right-hand side vector b
    b = u.copy()
    b[:-1] += lambd * (u[1:] - u[:-1])
    b[1:] += lambd * (u[:-1] - u[1:])

    # Solve the system of equations
    u = np.linalg.solve(A, b)

    return u

# MacCormack Scheme
def macCormack_method(a, T):
    dt = 0.5 * dx / np.abs(a)
    u = u0(x)  # Initial condition at t = 0
    for t in np.arange(0, T, dt):
        u = macCormack_method_update(u, a, dt, dx)

    return u

# Implement MacCormack discretization scheme
def macCormack_method_update(u, a, dt, dx):
    u_tilde = u.copy()

    # Predictor step
    for i in range(1, N - 1):
        u_tilde[i] = u[i] - a * dt / dx * (u[i + 1] - u[i])

    # Corrector step
    for i in range(1, N - 1):
        u[i] = 0.5 * (u[i] + u_tilde[i] - a * dt / dx * (u_tilde[i] - u_tilde[i - 1]))

    # Apply periodic boundary conditions
    u[0] = u[-2]
    u[-1] = u[1]

    return u

# Leapfrog Scheme
def leapfrog_method(a, T):
    dt = dx / (2 * np.abs(a))  # Adjusted time step for stability
    u_previous = u0(x - 0.5 * dt * a)  # Initial condition at n-1/2
    u_current = u0(x)  # Initial condition at n
    for t in np.arange(0, T, dt):
        u_next = leapfrog_method_update(u_previous, u_current, a, dt, dx)
        u_previous, u_current = u_current, u_next

    return u_current

# Implement Leapfrog discretization scheme
def leapfrog_method_update(u_previous, u_current, a, dt, dx):
    u_next = np.zeros_like(u_current)
    alpha = a * dt / dx

    # Update interior points
    for i in range(1, N - 1):
        u_next[i] = u_previous[i] - alpha * (u_current[i + 1] - u_current[i - 1])

    # Apply periodic boundary conditions
    u_next[0] = u_previous[0] - alpha * (u_current[1] - u_current[-2])
    u_next[-1] = u_next[0]

    return u_next

# Lax Scheme
def lax_method(a, T):
    dt = 0.5 * dx / np.abs(a)  # Adjusted time step for stability
    u = u0(x)  # Initial condition at t = 0
    for t in np.arange(0, T, dt):
        u = lax_method_update(u, a, dt, dx)

    return u

# Implement Lax discretization scheme
def lax_method_update(u, a, dt, dx):
    u_new = np.zeros_like(u)
    alpha = a * dt / (2 * dx)

    for i in range(1, N - 1):
        u_new[i] = 0.5 * (u[i + 1] + u[i - 1]) - alpha * (u[i + 1] - u[i - 1])

    # Apply periodic boundary conditions
    u_new[0] = u_new[-2]
    u_new[-1] = u_new[1]

    return u_new

# Lax-Wendroff Scheme
def lax_wendroff_method(a, T):
    dt = 0.5 * dx / a
    u = u0(x)  # Initial condition at t = 0
    for t in np.arange(0, T, dt):
        u = lax_wendroff_method_update(u, a, dt, dx)

    return u

# Implement Lax-Wendroff discretization scheme
def lax_wendroff_method_update(u, a, dt, dx):
    u_new = np.zeros_like(u)
    C = a * dt / dx
    for i in range(1, N - 1):
        u_new[i] = u[i] - 0.5 * C * (u[i + 1] - u[i - 1]) + 0.5 * C**2 * (u[i + 1] - 2 * u[i] + u[i - 1])

    # Apply periodic boundary conditions
    u_new[0] = u_new[-2]
    u_new[-1] = u_new[1]

    return u_new

# Upwind Scheme
def upwind_method(a, T):
    dt = 0.5 * dx / a
    u = u0(x)  # Initial condition at t = 0
    for t in np.arange(0, T, dt):
        u = upwind_method_update(u, a, dt, dx)

    return u

# Implement upwind discretization scheme
def upwind_method_update(u, a, dt, dx):
    u_new = np.zeros_like(u)
    for i in range(1, N - 1):
        if a > 0:
            u_new[i] = u[i] - a * dt / dx * (u[i] - u[i - 1])
        else:
            u_new[i] = u[i] - a * dt / dx * (u[i + 1] - u[i])

    # Apply periodic boundary conditions
    u_new[0] = u_new[-2]
    u_new[-1] = u_new[1]

    return u_new

# Exact Solution
def exact_solution(a, T):
    return u0(x - a * T)


# Function to update the plot for a specific method
def update(frame, u, line, a, dt, dx, method_update):
    t = frame * dt
    if t <= T:
        u[:] = method_update(u, a, dt, dx)
        line.set_ydata(u)
        ax.set_title(f'Method: {method_name}, Time: {t:.2f}')
    else:
        animation.event_source.stop()

# Function to create animation for a specific method
def create_animation(ax, u0, method_update, method_name):
    u = u0(x)
    line, = ax.plot(x, u, label=method_name)
    ax.set_xlabel("x")
    ax.set_ylabel("u(x, t)")
    ax.legend()
    animation = FuncAnimation(fig, update, frames=int(T/dt) + 1, fargs=(u, line, a, dt, dx, method_update), interval=50, blit=False)
    return animation

# Create initial plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(0, L)
ax.set_ylim(-1.5, 1.5)
ax.set_title("Wave Motion Animation for Different Discretization Schemes")

# List of discretization methods
methods = [
    ("Leapfrog", leapfrog_method_update),
    ("Lax-Wendroff", lax_wendroff_method_update),
    ("Upwind", upwind_method_update),
    ("MacCormack", macCormack_method_update),
    ("Lax", lax_method_update),
    ("Crank-Nicolson", crank_nicolson_method_update),
]

# Create animations for each method
animations = []
for method_name, method_update in methods:
    animation = create_animation(ax, u0, method_update, method_name)
    animations.append(animation)

# Show the animations
plt.show()
