import numpy as np

def compute_quintic_coefficients(q0, qd0, qdd0, q1, qd1, qdd1, T):
    # Time matrix (6x6)
    T_mat = np.array([
        [1,    0,     0,       0,        0,       0],
        [0,    1,     0,       0,        0,       0],
        [0,    0,     2,       0,        0,       0],
        [1,    T,     T**2,    T**3,     T**4,    T**5],
        [0,    1,     2*T,     3*T**2,   4*T**3,  5*T**4],
        [0,    0,     2,       6*T,     12*T**2, 20*T**3],
    ])
    
    # Stack conditions: [q0, qd0, qdd0, q1, qd1, qdd1]
    S = np.vstack([q0, qd0, qdd0, q1, qd1, qdd1])  # shape (6, 6)
    
    # Solve for all joints at once: (6x6) @ (6xN) = (6xN) => transpose result
    coeffs = np.linalg.solve(T_mat, S).T  # shape (6 joints, 6 coeffs)
    return coeffs  # Each row is [a0, a1, a2, a3, a4, a5]

# Inputs
q0 = [0, 0, 0, 0, 0, 0]
qd0 = [0, 0, 0, 0, 0, 0]
qdd0 = [0, 0, 0, 0, 0, 0]
q1 = [0.0, -1.57, -1.57, 3.14, -1.57, 0.0]
qd1 = [0, 0, 0, 0, 0, 0]
qdd1 = [0, 0, 0, 0, 0, 0]
T = 6.0

# Compute coefficients
coeffs = compute_quintic_coefficients(q0, qd0, qdd0, q1, qd1, qdd1, T)
print("Quintic coefficients per joint:")
print(coeffs)
# Round to 4 decimal places (you can change 4 to whatever you need)
rounded_coeffs = np.round(coeffs, 4)

print("Rounded coefficients:\n", rounded_coeffs)
