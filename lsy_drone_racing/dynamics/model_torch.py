"""PyTorch implementation of the SO(3) RPY drone dynamics model.

This module provides PyTorch-compatible dynamics functions for use with
MPPI and other gradient-based optimization methods.

Includes:
- Rotor dynamics (first-order thrust response)
- Aerodynamic drag model
- Second-order RPY attitude dynamics

Fixes applied:
- Proper Euler acceleration to angular acceleration conversion with Ẇ term
- Consistent disturbance torque handling in both Euler and quaternion paths
- Self-contained NumPy utilities for obs/state conversion
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import numpy as np
from scipy.spatial.transform import Rotation as R

if TYPE_CHECKING: 
    from typing import Optional, Union


# =============================================================================
# NumPy utility functions for obs/state conversion (self-contained)
# =============================================================================

def _ang_vel2rpy_rates_np(quat: np.ndarray, ang_vel:  np.ndarray) -> np.ndarray:
    """Convert angular velocity to RPY rates (NumPy version).
    
    Args: 
        quat:  Quaternion array of shape (..., 4) in xyzw format
        ang_vel: Angular velocity array of shape (..., 3)
        
    Returns:
        RPY rates array of shape (..., 3)
    """
    rot = R.from_quat(quat)
    euler = rot.as_euler("xyz")
    
    # Handle both single and batched inputs
    if euler.ndim == 1:
        roll, pitch = euler[0], euler[1]
        p, q, r = ang_vel[0], ang_vel[1], ang_vel[2]
    else:
        roll, pitch = euler[..., 0], euler[..., 1]
        p, q, r = ang_vel[..., 0], ang_vel[..., 1], ang_vel[..., 2]
    
    cp = np.cos(pitch)
    sp = np.sin(pitch)
    cr = np.cos(roll)
    sr = np.sin(roll)
    
    # Avoid singularity at pitch = ±90°
    cp = np.clip(cp, 1e-6, None)
    
    roll_rate = p + sr * sp / cp * q + cr * sp / cp * r
    pitch_rate = cr * q - sr * r
    yaw_rate = sr / cp * q + cr / cp * r
    
    return np.stack([roll_rate, pitch_rate, yaw_rate], axis=-1)


def _rpy_rates2ang_vel_np(quat: np.ndarray, rpy_rates: np.ndarray) -> np.ndarray:
    """Convert RPY rates to angular velocity (NumPy version).
    
    Args: 
        quat:  Quaternion array of shape (..., 4) in xyzw format
        rpy_rates: RPY rates array of shape (..., 3)
        
    Returns:
        Angular velocity array of shape (..., 3)
    """
    rot = R.from_quat(quat)
    euler = rot.as_euler("xyz")
    
    # Handle both single and batched inputs
    if euler.ndim == 1:
        roll, pitch = euler[0], euler[1]
        roll_rate, pitch_rate, yaw_rate = rpy_rates[0], rpy_rates[1], rpy_rates[2]
    else:
        roll, pitch = euler[..., 0], euler[..., 1]
        roll_rate, pitch_rate, yaw_rate = rpy_rates[..., 0], rpy_rates[..., 1], rpy_rates[..., 2]
    
    cp = np.cos(pitch)
    sp = np.sin(pitch)
    cr = np.cos(roll)
    sr = np.sin(roll)
    
    p = roll_rate - sp * yaw_rate
    q = cr * pitch_rate + sr * cp * yaw_rate
    r = -sr * pitch_rate + cr * cp * yaw_rate
    
    return np.stack([p, q, r], axis=-1)


# =============================================================================
# PyTorch rotation utilities
# =============================================================================

def quat_to_rotation_matrix(quat: torch.Tensor) -> torch.Tensor:
    """Convert quaternion (xyzw) to rotation matrix.
    
    Args: 
        quat:  Quaternion tensor of shape (..., 4) in xyzw format
        
    Returns:
        Rotation matrix of shape (..., 3, 3)
    """
    # Normalize quaternion
    quat = quat / torch.norm(quat, dim=-1, keepdim=True)
    
    x, y, z, w = torch.split(quat, 1, dim=-1)
    
    # Compute rotation matrix elements
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    xw = x * w
    yw = y * w
    zw = z * w
    
    R00 = 1 - 2 * (yy + zz)
    R01 = 2 * (xy - zw)
    R02 = 2 * (xz + yw)
    R10 = 2 * (xy + zw)
    R11 = 1 - 2 * (xx + zz)
    R12 = 2 * (yz - xw)
    R20 = 2 * (xz - yw)
    R21 = 2 * (yz + xw)
    R22 = 1 - 2 * (xx + yy)
    
    row1 = torch.cat([R00, R01, R02], dim=-1)
    row2 = torch.cat([R10, R11, R12], dim=-1)
    row3 = torch.cat([R20, R21, R22], dim=-1)
    
    rot_matrix = torch.stack([row1, row2, row3], dim=-2)
    
    return rot_matrix


def quat_to_euler(quat: torch.Tensor) -> torch.Tensor:
    """Convert quaternion (xyzw) to Euler angles (xyz/roll-pitch-yaw).
    
    Args:
        quat: Quaternion tensor of shape (..., 4) in xyzw format
        
    Returns: 
        Euler angles tensor of shape (..., 3) in xyz order
    """
    # Normalize quaternion
    quat = quat / torch.norm(quat, dim=-1, keepdim=True)
    
    x, y, z, w = torch.split(quat, 1, dim=-1)
    
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)
    
    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    # Clamp to avoid numerical issues at poles
    sinp = torch.clamp(sinp, -1.0, 1.0)
    pitch = torch.asin(sinp)
    
    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)
    
    return torch.cat([roll, pitch, yaw], dim=-1)


def euler_to_rotation_matrix(euler:  torch.Tensor) -> torch.Tensor:
    """Convert Euler angles (xyz) to rotation matrix.
    
    Args: 
        euler:  Euler angles tensor of shape (..., 3) in xyz order
        
    Returns: 
        Rotation matrix of shape (..., 3, 3)
    """
    roll, pitch, yaw = torch.split(euler, 1, dim=-1)
    
    # Compute trig values
    cr = torch.cos(roll)
    sr = torch.sin(roll)
    cp = torch.cos(pitch)
    sp = torch.sin(pitch)
    cy = torch.cos(yaw)
    sy = torch.sin(yaw)
    
    # Rotation matrix for ZYX Euler angles
    R00 = cy * cp
    R01 = cy * sp * sr - sy * cr
    R02 = cy * sp * cr + sy * sr
    R10 = sy * cp
    R11 = sy * sp * sr + cy * cr
    R12 = sy * sp * cr - cy * sr
    R20 = -sp
    R21 = cp * sr
    R22 = cp * cr
    
    row1 = torch.cat([R00, R01, R02], dim=-1)
    row2 = torch.cat([R10, R11, R12], dim=-1)
    row3 = torch.cat([R20, R21, R22], dim=-1)
    
    rot_matrix = torch.stack([row1, row2, row3], dim=-2)
    
    return rot_matrix


# =============================================================================
# Euler <-> Body rate conversions (with proper Jacobian handling)
# =============================================================================

def get_euler_rate_matrix(euler: torch.Tensor) -> torch.Tensor:
    """Compute the transformation matrix W from Euler rates to body angular velocity.
    
    ω = W(rpy) · ṙpy
    
    W = [[1,    0,      -sin(pitch)         ],
         [0,    cos(roll),  sin(roll)*cos(pitch) ],
         [0,   -sin(roll),  cos(roll)*cos(pitch) ]]
    
    Args:
        euler: Euler angles tensor of shape (..., 3)
        
    Returns: 
        W matrix of shape (..., 3, 3)
    """
    roll, pitch, _ = torch.split(euler, 1, dim=-1)
    
    cr = torch.cos(roll)
    sr = torch.sin(roll)
    cp = torch.cos(pitch)
    sp = torch.sin(pitch)
    
    zero = torch.zeros_like(roll)
    one = torch.ones_like(roll)
    
    # Build W matrix
    row1 = torch.cat([one, zero, -sp], dim=-1)
    row2 = torch.cat([zero, cr, sr * cp], dim=-1)
    row3 = torch.cat([zero, -sr, cr * cp], dim=-1)
    
    W = torch.stack([row1, row2, row3], dim=-2)
    
    return W


def get_euler_rate_matrix_inv(euler: torch.Tensor) -> torch.Tensor:
    """Compute the inverse transformation matrix W^{-1} from body angular velocity to Euler rates.
    
    ṙpy = W^{-1}(rpy) · ω
    
    Args:
        euler:  Euler angles tensor of shape (..., 3)
        
    Returns:
        W_inv matrix of shape (..., 3, 3)
    """
    roll, pitch, _ = torch.split(euler, 1, dim=-1)
    
    cr = torch.cos(roll)
    sr = torch.sin(roll)
    cp = torch.cos(pitch)
    sp = torch.sin(pitch)
    tp = torch.tan(pitch)
    
    # Avoid singularity
    cp = torch.clamp(torch.abs(cp), min=1e-6) * torch.sign(cp + 1e-10)
    
    zero = torch.zeros_like(roll)
    one = torch.ones_like(roll)
    
    # Build W_inv matrix
    row1 = torch.cat([one, sr * tp, cr * tp], dim=-1)
    row2 = torch.cat([zero, cr, -sr], dim=-1)
    row3 = torch.cat([zero, sr / cp, cr / cp], dim=-1)
    
    W_inv = torch.stack([row1, row2, row3], dim=-2)
    
    return W_inv


def get_euler_rate_matrix_deriv(
    euler: torch.Tensor,
    rpy_rates:  torch.Tensor,
) -> torch.Tensor:
    """Compute the time derivative of the W matrix:  Ẇ(rpy, ṙpy).
    
    Ẇ = ∂W/∂roll · roll_rate + ∂W/∂pitch · pitch_rate
    
    Args: 
        euler:  Euler angles tensor of shape (..., 3)
        rpy_rates:  Euler rates tensor of shape (..., 3)
        
    Returns:
        W_dot matrix of shape (..., 3, 3)
    """
    roll, pitch, _ = torch.split(euler, 1, dim=-1)
    roll_rate, pitch_rate, _ = torch.split(rpy_rates, 1, dim=-1)
    
    cr = torch.cos(roll)
    sr = torch.sin(roll)
    cp = torch.cos(pitch)
    sp = torch.sin(pitch)
    
    zero = torch.zeros_like(roll)
    
    # ∂W/∂roll: 
    # [[0, 0, 0],
    #  [0, -sin(roll), cos(roll)*cos(pitch)],
    #  [0, -cos(roll), -sin(roll)*cos(pitch)]]
    
    # ∂W/∂pitch: 
    # [[0, 0, -cos(pitch)],
    #  [0, 0, -sin(roll)*sin(pitch)],
    #  [0, 0, -cos(roll)*sin(pitch)]]
    
    # Ẇ = ∂W/∂roll * roll_rate + ∂W/∂pitch * pitch_rate
    W_dot_00 = zero
    W_dot_01 = zero
    W_dot_02 = -cp * pitch_rate
    
    W_dot_10 = zero
    W_dot_11 = -sr * roll_rate
    W_dot_12 = cr * cp * roll_rate - sr * sp * pitch_rate
    
    W_dot_20 = zero
    W_dot_21 = -cr * roll_rate
    W_dot_22 = -sr * cp * roll_rate - cr * sp * pitch_rate
    
    row1 = torch.cat([W_dot_00, W_dot_01, W_dot_02], dim=-1)
    row2 = torch.cat([W_dot_10, W_dot_11, W_dot_12], dim=-1)
    row3 = torch.cat([W_dot_20, W_dot_21, W_dot_22], dim=-1)
    
    W_dot = torch.stack([row1, row2, row3], dim=-2)
    
    return W_dot


def rpy_rates_to_ang_vel(euler: torch.Tensor, rpy_rates: torch.Tensor) -> torch.Tensor:
    """Convert RPY rates to angular velocity.
    
    ω = W(rpy) · ṙpy
    
    Args:
        euler: Euler angles tensor of shape (..., 3)
        rpy_rates: RPY rates tensor of shape (..., 3)
        
    Returns: 
        Angular velocity tensor of shape (..., 3)
    """
    W = get_euler_rate_matrix(euler)
    ang_vel = torch.matmul(W, rpy_rates.unsqueeze(-1)).squeeze(-1)
    return ang_vel


def ang_vel_to_rpy_rates(quat: torch.Tensor, ang_vel: torch.Tensor) -> torch.Tensor:
    """Convert angular velocity to RPY rates.
    
    ṙpy = W^{-1}(rpy) · ω
    
    Args:
        quat: Quaternion tensor of shape (..., 4) in xyzw format
        ang_vel: Angular velocity tensor of shape (..., 3)
        
    Returns:
        RPY rates tensor of shape (..., 3)
    """
    euler = quat_to_euler(quat)
    W_inv = get_euler_rate_matrix_inv(euler)
    rpy_rates = torch.matmul(W_inv, ang_vel.unsqueeze(-1)).squeeze(-1)
    return rpy_rates


def rpy_rates_deriv_to_ang_vel_deriv(
    euler: torch.Tensor,
    rpy_rates: torch.Tensor,
    rpy_rates_deriv: torch.Tensor,
) -> torch.Tensor:
    """Convert RPY acceleration to angular acceleration.
    
    Implements:  ω̇ = Ẇ(rpy, ṙpy) · ṙpy + W(rpy) · r̈py
    
    Args: 
        euler: Euler angles tensor of shape (..., 3)
        rpy_rates: RPY rates tensor of shape (..., 3)
        rpy_rates_deriv: RPY acceleration tensor of shape (..., 3)
        
    Returns: 
        Angular acceleration tensor of shape (..., 3)
    """
    W = get_euler_rate_matrix(euler)
    W_dot = get_euler_rate_matrix_deriv(euler, rpy_rates)
    
    # ω̇ = Ẇ · ṙpy + W · r̈py
    term1 = torch.matmul(W_dot, rpy_rates.unsqueeze(-1)).squeeze(-1)
    term2 = torch.matmul(W, rpy_rates_deriv.unsqueeze(-1)).squeeze(-1)
    
    ang_vel_deriv = term1 + term2
    
    return ang_vel_deriv


def ang_vel_deriv_to_rpy_rates_deriv(
    euler:  torch.Tensor,
    rpy_rates: torch.Tensor,
    ang_vel_deriv: torch.Tensor,
) -> torch.Tensor:
    """Convert angular acceleration to RPY acceleration.
    
    Inverts: ω̇ = Ẇ · ṙpy + W · r̈py
    Solves: r̈py = W^{-1} · (ω̇ - Ẇ · ṙpy)
    
    Args: 
        euler:  Euler angles tensor of shape (..., 3)
        rpy_rates: RPY rates tensor of shape (..., 3)
        ang_vel_deriv: Angular acceleration tensor of shape (..., 3)
        
    Returns:
        RPY acceleration tensor of shape (..., 3)
    """
    W_inv = get_euler_rate_matrix_inv(euler)
    W_dot = get_euler_rate_matrix_deriv(euler, rpy_rates)
    
    # r̈py = W^{-1} · (ω̇ - Ẇ · ṙpy)
    W_dot_rpy_rates = torch.matmul(W_dot, rpy_rates.unsqueeze(-1)).squeeze(-1)
    rpy_rates_deriv = torch.matmul(
        W_inv, (ang_vel_deriv - W_dot_rpy_rates).unsqueeze(-1)
    ).squeeze(-1)
    
    return rpy_rates_deriv


# =============================================================================
# Drag computation
# =============================================================================

def compute_drag_acceleration(
    vel: torch.Tensor,
    rot: torch.Tensor,
    drag_matrix: torch.Tensor,
    mass: float,
) -> torch.Tensor:
    """Compute acceleration due to aerodynamic drag.
    
    The drag is computed in the body frame and then transformed to the world frame.
    drag_acc = (1/m) * R * D * R^T * v
    
    where R is the rotation matrix (body to world), D is the drag coefficient matrix,
    and v is the velocity in the world frame.
    
    Args:
        vel: Velocity tensor of shape (..., 3) in world frame
        rot:  Rotation matrix of shape (..., 3, 3) from body to world
        drag_matrix:  Drag coefficient matrix of shape (3, 3)
        mass: Mass of the drone (kg)
        
    Returns:
        Drag acceleration tensor of shape (..., 3) in world frame
    """
    # rot is body-to-world, so rot.T is world-to-body
    rot_inv = rot.transpose(-2, -1)  # world to body
    
    # Transform velocity to body frame
    vel_body = torch.matmul(rot_inv, vel.unsqueeze(-1))  # (..., 3, 1)
    
    # Apply drag in body frame
    drag_body = torch.matmul(drag_matrix, vel_body)  # (..., 3, 1)
    
    # Transform drag back to world frame
    drag_world = torch.matmul(rot, drag_body).squeeze(-1)  # (..., 3)
    
    # Divide by mass to get acceleration
    drag_acc = drag_world / mass
    
    return drag_acc


# =============================================================================
# Disturbance torque handling utilities
# =============================================================================

def apply_disturbance_torque_body_frame(
    ang_vel: torch.Tensor,
    ang_vel_dot: torch.Tensor,
    dist_t_body: torch.Tensor,
    J:  torch.Tensor,
    J_inv: torch.Tensor,
) -> torch.Tensor:
    """Apply disturbance torque in body frame and return updated angular acceleration.
    
    Uses Euler's rotation equations: 
    τ = J · ω̇ + ω × (J · ω)
    
    Args:
        ang_vel: Angular velocity in body frame, shape (..., 3)
        ang_vel_dot:  Angular acceleration in body frame, shape (..., 3)
        dist_t_body:  Disturbance torque in body frame, shape (..., 3)
        J: Inertia matrix, shape (3, 3)
        J_inv: Inverse inertia matrix, shape (3, 3)
        
    Returns:
        Updated angular acceleration, shape (..., 3)
    """
    # Compute current torque from dynamics
    # τ = J · ω̇ + ω × (J · ω)
    J_omega = torch.matmul(J, ang_vel.unsqueeze(-1)).squeeze(-1)
    gyroscopic_term = torch.cross(ang_vel, J_omega, dim=-1)
    
    torque = torch.matmul(J, ang_vel_dot.unsqueeze(-1)).squeeze(-1) + gyroscopic_term
    
    # Add disturbance torque
    torque = torque + dist_t_body
    
    # Convert back to angular acceleration
    # ω̇ = J^{-1} · (τ - ω × (J · ω))
    ang_vel_dot_new = torch.matmul(J_inv, (torque - gyroscopic_term).unsqueeze(-1)).squeeze(-1)
    
    return ang_vel_dot_new


# =============================================================================
# Dynamics functions
# =============================================================================

def dynamics_euler(
    state: torch.Tensor,
    cmd: torch.Tensor,
    dt:  float,
    mass: float,
    gravity_vec: torch.Tensor,
    J: torch.Tensor,
    J_inv: torch.Tensor,
    acc_coef: torch.Tensor,
    cmd_f_coef: torch.Tensor,
    rpy_coef: torch.Tensor,
    rpy_rates_coef: torch.Tensor,
    cmd_rpy_coef:  torch.Tensor,
    dist_f:  Optional[torch.Tensor] = None,
    dist_t: Optional[torch.Tensor] = None,
    thrust_time_coef: Optional[torch.Tensor] = None,
    drag_matrix: Optional[torch.Tensor] = None,
    model_rotor_dynamics: bool = False,
) -> torch.Tensor:
    """Fitted model with linear, second order RPY dynamics (Euler representation).
    
    This is the PyTorch version of the dynamics suitable for MPPI.
    State is in Euler angle representation for direct attitude control.
    
    Args:
        state: State tensor of shape (..., 12) or (..., 13) if rotor dynamics enabled: 
            [pos (3), rpy (3), vel (3), drpy (3), (rotor_thrust (1) if enabled)]
        cmd: Command tensor of shape (..., 4) containing: 
            [roll_cmd, pitch_cmd, yaw_cmd, thrust_cmd]
        dt: Time step for integration (s)
        mass: Mass of the drone (kg)
        gravity_vec:  Gravity vector (m/s^2), shape (3,)
        J: Inertia matrix (kg m^2), shape (3, 3)
        J_inv: Inverse inertia matrix, shape (3, 3)
        acc_coef:  Coefficient for acceleration (1/s^2), scalar or shape (1,)
        cmd_f_coef: Coefficient for collective thrust (N/rad^2), scalar or shape (1,)
        rpy_coef: Coefficient for RPY dynamics (1/s), shape (3,)
        rpy_rates_coef: Coefficient for RPY rates dynamics (1/s^2), shape (3,)
        cmd_rpy_coef: Coefficient for RPY command dynamics (1/s), shape (3,)
        dist_f: Disturbance force (N) in world frame, optional
        dist_t:  Disturbance torque (Nm) in world frame, optional
        thrust_time_coef: Time constant for rotor dynamics (s), optional
        drag_matrix:  Drag coefficient matrix (3, 3), optional
        model_rotor_dynamics: If True, include rotor dynamics in state
        
    Returns: 
        Next state tensor of shape (..., 12) or (..., 13)
    """
    # Extract state components
    pos = state[..., 0:3]
    rpy = state[..., 3:6]
    vel = state[..., 6:9]
    drpy = state[..., 9:12]
    
    # Extract rotor thrust state if modeling rotor dynamics
    if model_rotor_dynamics:
        rotor_thrust = state[..., 12:13]
    else: 
        rotor_thrust = None
    
    # Extract command components
    cmd_rpy = cmd[..., 0:3]
    cmd_thrust = cmd[..., 3:4]
    
    # Compute rotation matrix from Euler angles
    rot = euler_to_rotation_matrix(rpy)
    
    # Compute actual thrust (with or without rotor dynamics)
    if model_rotor_dynamics and rotor_thrust is not None:
        # First-order rotor dynamics:  actual thrust lags behind commanded thrust
        rotor_thrust_dot = (cmd_thrust - rotor_thrust) / thrust_time_coef
        actual_thrust = rotor_thrust
    else:
        # Direct thrust application (no dynamics)
        actual_thrust = cmd_thrust
        rotor_thrust_dot = None
    
    # Compute thrust magnitude
    thrust = acc_coef + cmd_f_coef * actual_thrust
    
    # Thrust vector in body frame (z-axis)
    thrust_body = torch.zeros_like(vel)
    thrust_body[..., 2] = thrust.squeeze(-1)
    
    # Transform thrust to world frame
    thrust_world = torch.matmul(rot, thrust_body.unsqueeze(-1)).squeeze(-1)
    
    # Linear dynamics
    pos_dot = vel
    vel_dot = thrust_world / mass + gravity_vec
    
    # Add drag if drag_matrix is provided
    if drag_matrix is not None:
        drag_acc = compute_drag_acceleration(vel, rot, drag_matrix, mass)
        vel_dot = vel_dot + drag_acc
    
    if dist_f is not None:
        vel_dot = vel_dot + dist_f / mass
    
    # Rotational dynamics (second-order linear model)
    rpy_dot = drpy
    ddrpy = rpy_coef * rpy + rpy_rates_coef * drpy + cmd_rpy_coef * cmd_rpy
    
    # If torque disturbances are present, handle properly with Jacobian
    if dist_t is not None:
        # Convert Euler quantities to body frame
        ang_vel = rpy_rates_to_ang_vel(rpy, drpy)
        ang_vel_dot = rpy_rates_deriv_to_ang_vel_deriv(rpy, drpy, ddrpy)
        
        # Transform disturbance torque from world to body frame
        rot_inv = rot.transpose(-2, -1)
        dist_t_body = torch.matmul(rot_inv, dist_t.unsqueeze(-1)).squeeze(-1)
        
        # Apply disturbance torque in body frame
        ang_vel_dot_new = apply_disturbance_torque_body_frame(
            ang_vel, ang_vel_dot, dist_t_body, J, J_inv
        )
        
        # Convert back to Euler acceleration
        ddrpy = ang_vel_deriv_to_rpy_rates_deriv(rpy, drpy, ang_vel_dot_new)
    
    # Integrate state (Euler integration)
    pos_next = pos + pos_dot * dt
    vel_next = vel + vel_dot * dt
    rpy_next = rpy + rpy_dot * dt
    drpy_next = drpy + ddrpy * dt
    
    # Wrap angles to [-pi, pi]
    rpy_next = torch.atan2(torch.sin(rpy_next), torch.cos(rpy_next))
    
    # Concatenate next state
    if model_rotor_dynamics and rotor_thrust_dot is not None: 
        rotor_thrust_next = rotor_thrust + rotor_thrust_dot * dt
        state_next = torch.cat(
            [pos_next, rpy_next, vel_next, drpy_next, rotor_thrust_next], dim=-1
        )
    else:
        state_next = torch.cat([pos_next, rpy_next, vel_next, drpy_next], dim=-1)
    
    return state_next


def dynamics_quat(
    state: torch.Tensor,
    cmd:  torch.Tensor,
    dt: float,
    mass:  float,
    gravity_vec: torch.Tensor,
    J: torch.Tensor,
    J_inv: torch.Tensor,
    acc_coef:  torch.Tensor,
    cmd_f_coef: torch.Tensor,
    rpy_coef: torch.Tensor,
    rpy_rates_coef: torch.Tensor,
    cmd_rpy_coef: torch.Tensor,
    dist_f: Optional[torch.Tensor] = None,
    dist_t: Optional[torch.Tensor] = None,
    thrust_time_coef: Optional[torch.Tensor] = None,
    drag_matrix: Optional[torch.Tensor] = None,
    model_rotor_dynamics: bool = False,
) -> torch.Tensor:
    """Fitted model with quaternion state representation.
    
    This version uses quaternions for the state but still applies the
    second-order linear RPY dynamics model.
    
    Args:
        state: State tensor of shape (..., 13) or (..., 14) if rotor dynamics enabled:
            [pos (3), quat (4, xyzw), vel (3), ang_vel (3), (rotor_thrust (1) if enabled)]
        cmd: Command tensor of shape (..., 4) containing:
            [roll_cmd, pitch_cmd, yaw_cmd, thrust_cmd]
        dt:  Time step for integration (s)
        mass: Mass of the drone (kg)
        gravity_vec:  Gravity vector (m/s^2), shape (3,)
        J: Inertia matrix (kg m^2), shape (3, 3)
        J_inv: Inverse inertia matrix, shape (3, 3)
        acc_coef: Coefficient for acceleration
        cmd_f_coef: Coefficient for collective thrust
        rpy_coef: Coefficient for RPY dynamics
        rpy_rates_coef: Coefficient for RPY rates dynamics
        cmd_rpy_coef:  Coefficient for RPY command dynamics
        dist_f: Disturbance force (optional)
        dist_t: Disturbance torque (optional)
        thrust_time_coef: Time constant for rotor dynamics (s), optional
        drag_matrix: Drag coefficient matrix (3, 3), optional
        model_rotor_dynamics: If True, include rotor dynamics in state
        
    Returns:
        Next state tensor of shape (..., 13) or (..., 14)
    """
    # Extract state components
    pos = state[..., 0:3]
    quat = state[..., 3:7]
    vel = state[..., 7:10]
    ang_vel = state[..., 10:13]
    
    # Extract rotor thrust state if modeling rotor dynamics
    if model_rotor_dynamics:
        rotor_thrust = state[..., 13:14]
    else:
        rotor_thrust = None
    
    # Convert to Euler representation for the RPY dynamics model
    rpy = quat_to_euler(quat)
    drpy = ang_vel_to_rpy_rates(quat, ang_vel)
    
    # Extract command components
    cmd_rpy = cmd[..., 0:3]
    cmd_thrust = cmd[..., 3:4]
    
    # Compute rotation matrix
    rot = quat_to_rotation_matrix(quat)
    
    # Compute actual thrust (with or without rotor dynamics)
    if model_rotor_dynamics and rotor_thrust is not None:
        # First-order rotor dynamics
        rotor_thrust_dot = (cmd_thrust - rotor_thrust) / thrust_time_coef
        actual_thrust = rotor_thrust
    else: 
        actual_thrust = cmd_thrust
        rotor_thrust_dot = None
    
    # Compute thrust magnitude
    thrust = acc_coef + cmd_f_coef * actual_thrust
    
    # Thrust vector in body frame (z-axis)
    thrust_body = torch.zeros_like(vel)
    thrust_body[..., 2] = thrust.squeeze(-1)
    
    # Transform thrust to world frame
    thrust_world = torch.matmul(rot, thrust_body.unsqueeze(-1)).squeeze(-1)
    
    # Linear dynamics
    pos_dot = vel
    vel_dot = thrust_world / mass + gravity_vec
    
    # Add drag if drag_matrix is provided
    if drag_matrix is not None:
        drag_acc = compute_drag_acceleration(vel, rot, drag_matrix, mass)
        vel_dot = vel_dot + drag_acc
    
    if dist_f is not None: 
        vel_dot = vel_dot + dist_f / mass
    
    # Rotational dynamics using second-order RPY model
    ddrpy = rpy_coef * rpy + rpy_rates_coef * drpy + cmd_rpy_coef * cmd_rpy
    
    # Convert RPY acceleration to angular acceleration (with proper Jacobian)
    ang_vel_dot = rpy_rates_deriv_to_ang_vel_deriv(rpy, drpy, ddrpy)
    
    if dist_t is not None:
        # Transform disturbance torque from world to body frame
        rot_inv = rot.transpose(-2, -1)
        dist_t_body = torch.matmul(rot_inv, dist_t.unsqueeze(-1)).squeeze(-1)
        
        # Apply disturbance torque in body frame
        ang_vel_dot = apply_disturbance_torque_body_frame(
            ang_vel, ang_vel_dot, dist_t_body, J, J_inv
        )
    
    # Quaternion derivative using quaternion kinematics
    # q̇ = 0.5 * Ω(ω) * q
    wx, wy, wz = torch.split(ang_vel, 1, dim=-1)
    zero = torch.zeros_like(wx)
    
    # Quaternion rate matrix (Hamilton convention for xyzw quaternion)
    omega_matrix = torch.cat([
        torch.cat([zero, -wx, -wy, -wz], dim=-1).unsqueeze(-2),
        torch.cat([wx, zero, wz, -wy], dim=-1).unsqueeze(-2),
        torch.cat([wy, -wz, zero, wx], dim=-1).unsqueeze(-2),
        torch.cat([wz, wy, -wx, zero], dim=-1).unsqueeze(-2),
    ], dim=-2)
    
    quat_dot = 0.5 * torch.matmul(omega_matrix, quat.unsqueeze(-1)).squeeze(-1)
    
    # Integrate state (Euler integration)
    pos_next = pos + pos_dot * dt
    vel_next = vel + vel_dot * dt
    quat_next = quat + quat_dot * dt
    ang_vel_next = ang_vel + ang_vel_dot * dt
    
    # Normalize quaternion to maintain unit norm
    quat_next = quat_next / torch.norm(quat_next, dim=-1, keepdim=True)
    
    # Concatenate next state
    if model_rotor_dynamics and rotor_thrust_dot is not None:
        rotor_thrust_next = rotor_thrust + rotor_thrust_dot * dt
        state_next = torch.cat(
            [pos_next, quat_next, vel_next, ang_vel_next, rotor_thrust_next], dim=-1
        )
    else: 
        state_next = torch.cat([pos_next, quat_next, vel_next, ang_vel_next], dim=-1)
    
    return state_next


# =============================================================================
# DroneModelTorch class
# =============================================================================

class DroneModelTorch: 
    """PyTorch wrapper for drone dynamics with parameter management."""
    
    def __init__(
        self,
        parameters: dict,
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.float64,
        use_euler_state: bool = True,
        model_rotor_dynamics:  bool = False,
        model_drag: bool = False,
    ):
        """Initialize the PyTorch drone model.
        
        Args:
            parameters:  Dictionary containing model parameters: 
                - mass:  float, drone mass (kg)
                - gravity_vec: array-like (3,), gravity vector (m/s^2)
                - J: array-like (3, 3), inertia matrix (kg m^2)
                - J_inv: array-like (3, 3), inverse inertia matrix
                - acc_coef: float or array-like, acceleration coefficient
                - cmd_f_coef: float or array-like, thrust command coefficient
                - rpy_coef:  array-like (3,), RPY dynamics coefficient
                - rpy_rates_coef: array-like (3,), RPY rates dynamics coefficient
                - cmd_rpy_coef: array-like (3,), RPY command coefficient
                - thrust_time_coef: float, rotor time constant (required if model_rotor_dynamics)
                - drag_matrix: array-like (3, 3), drag coefficient matrix (required if model_drag)
                - thrust_min: float, optional, minimum thrust (default 0.0)
                - thrust_max: float, optional, maximum thrust (default 1.0)
                - rpy_max: float, optional, max RPY command in radians (default 0.5)
            device: Device to place tensors on
            dtype: Data type for tensors
            use_euler_state: If True, use Euler state (12D/13D), else quaternion (13D/14D)
            model_rotor_dynamics: If True, include rotor dynamics (adds 1D to state)
            model_drag: If True, include aerodynamic drag model
        """
        self.device = device
        self.dtype = dtype
        self.use_euler_state = use_euler_state
        self.model_rotor_dynamics = model_rotor_dynamics
        self.model_drag = model_drag
        
        # Convert parameters to tensors
        self.mass = float(parameters["mass"])
        self.gravity_vec = torch.tensor(
            parameters["gravity_vec"], dtype=dtype, device=device
        )
        self.J = torch.tensor(parameters["J"], dtype=dtype, device=device)
        self.J_inv = torch.tensor(parameters["J_inv"], dtype=dtype, device=device)
        self.acc_coef = torch.tensor(
            parameters["acc_coef"], dtype=dtype, device=device
        )
        self.cmd_f_coef = torch.tensor(
            parameters["cmd_f_coef"], dtype=dtype, device=device
        )
        self.rpy_coef = torch.tensor(
            parameters["rpy_coef"], dtype=dtype, device=device
        )
        self.rpy_rates_coef = torch.tensor(
            parameters["rpy_rates_coef"], dtype=dtype, device=device
        )
        self.cmd_rpy_coef = torch.tensor(
            parameters["cmd_rpy_coef"], dtype=dtype, device=device
        )
        
        # Rotor dynamics parameters
        if model_rotor_dynamics:
            if "thrust_time_coef" not in parameters:
                raise ValueError("thrust_time_coef required when model_rotor_dynamics=True")
            self.thrust_time_coef = torch.tensor(
                parameters["thrust_time_coef"], dtype=dtype, device=device
            )
        else:
            self.thrust_time_coef = None
        
        # Drag parameters
        if model_drag:
            if "drag_matrix" not in parameters: 
                raise ValueError("drag_matrix required when model_drag=True")
            self.drag_matrix = torch.tensor(
                parameters["drag_matrix"], dtype=dtype, device=device
            )
        else:
            self.drag_matrix = None
        
        # Control limits
        self.thrust_min = parameters.get("thrust_min", 0.0)
        self.thrust_max = parameters.get("thrust_max", 1.0)
        self.rpy_max = parameters.get("rpy_max", 0.5)  # radians
        
        # State dimensions
        # Base:  12 (Euler) or 13 (quat)
        # +1 if rotor dynamics enabled
        base_dim = 12 if use_euler_state else 13
        self.nx = base_dim + (1 if model_rotor_dynamics else 0)
        self.nu = 4  # [roll_cmd, pitch_cmd, yaw_cmd, thrust_cmd]
    
    def dynamics(
        self,
        state: torch.Tensor,
        cmd: torch.Tensor,
        dt: float,
        dist_f: Optional[torch.Tensor] = None,
        dist_t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute next state given current state and command.
        
        Args:
            state: Current state tensor
            cmd: Control command tensor
            dt: Time step (s)
            dist_f: Force disturbance in world frame (optional)
            dist_t: Torque disturbance in world frame (optional)
            
        Returns: 
            Next state tensor
        """
        if self.use_euler_state:
            return dynamics_euler(
                state, cmd, dt,
                self.mass, self.gravity_vec, self.J, self.J_inv,
                self.acc_coef, self.cmd_f_coef, self.rpy_coef,
                self.rpy_rates_coef, self.cmd_rpy_coef,
                dist_f, dist_t,
                thrust_time_coef=self.thrust_time_coef,
                drag_matrix=self.drag_matrix,
                model_rotor_dynamics=self.model_rotor_dynamics,
            )
        else:
            return dynamics_quat(
                state, cmd, dt,
                self.mass, self.gravity_vec, self.J, self.J_inv,
                self.acc_coef, self.cmd_f_coef, self.rpy_coef,
                self.rpy_rates_coef, self.cmd_rpy_coef,
                dist_f, dist_t,
                thrust_time_coef=self.thrust_time_coef,
                drag_matrix=self.drag_matrix,
                model_rotor_dynamics=self.model_rotor_dynamics,
            )
    
    def get_initial_rotor_state(
        self,
        batch_shape: tuple = (),
        initial_thrust: float = 0.0,
    ) -> torch.Tensor:
        """Get initial rotor thrust state for rotor dynamics.
        
        Args: 
            batch_shape:  Shape for batched states
            initial_thrust: Initial thrust value
            
        Returns: 
            Rotor thrust state tensor of shape (*batch_shape, 1)
        """
        if not self.model_rotor_dynamics: 
            raise ValueError("Rotor dynamics not enabled")
        
        return torch.full(
            (*batch_shape, 1),
            initial_thrust,
            dtype=self.dtype,
            device=self.device,
        )
    
    def obs_to_state(self, obs: dict) -> torch.Tensor:
        """Convert observation dictionary to state tensor.
        
        Args: 
            obs:  Observation dictionary from environment containing:
                - pos: position (3,)
                - quat: quaternion xyzw (4,)
                - vel: velocity (3,)
                - ang_vel: angular velocity (3,)
                - rotor_thrust:  (optional) rotor thrust state
                
        Returns:
            State tensor
        """
        if self.use_euler_state:
            # Convert quaternion to Euler angles
            quat = np.asarray(obs["quat"])
            rot = R.from_quat(quat)
            rpy = rot.as_euler("xyz")
            
            # Convert angular velocity to RPY rates (using local NumPy function)
            drpy = _ang_vel2rpy_rates_np(quat, np.asarray(obs["ang_vel"]))
            
            state_components = [
                np.asarray(obs["pos"]),
                rpy,
                np.asarray(obs["vel"]),
                drpy
            ]
            
            # Add rotor thrust state if modeling rotor dynamics
            if self.model_rotor_dynamics: 
                if "rotor_thrust" in obs: 
                    state_components.append(np.atleast_1d(obs["rotor_thrust"]))
                else:
                    # Initialize with zero
                    state_components.append(np.array([0.0]))
            
            state = np.concatenate(state_components)
        else:
            state_components = [
                np.asarray(obs["pos"]),
                np.asarray(obs["quat"]),
                np.asarray(obs["vel"]),
                np.asarray(obs["ang_vel"])
            ]
            
            # Add rotor thrust state if modeling rotor dynamics
            if self.model_rotor_dynamics: 
                if "rotor_thrust" in obs:
                    state_components.append(np.atleast_1d(obs["rotor_thrust"]))
                else: 
                    state_components.append(np.array([0.0]))
            
            state = np.concatenate(state_components)
        
        return torch.tensor(state, dtype=self.dtype, device=self.device)
    
    def state_to_obs(self, state: torch.Tensor) -> dict:
        """Convert state tensor to observation dictionary.
        
        Args:
            state:  State tensor
            
        Returns: 
            Observation dictionary with pos, quat, vel, ang_vel, and optionally
            rpy, drpy, rotor_thrust
        """
        state_np = state.detach().cpu().numpy()
        
        if self.use_euler_state:
            pos = state_np[..., 0:3]
            rpy = state_np[..., 3:6]
            vel = state_np[..., 6:9]
            drpy = state_np[..., 9:12]
            
            # Convert Euler to quaternion
            rot = R.from_euler("xyz", rpy)
            quat = rot.as_quat()
            
            # Convert RPY rates to angular velocity (using local NumPy function)
            ang_vel = _rpy_rates2ang_vel_np(quat, drpy)
            
            obs = {
                "pos": pos,
                "quat": quat,
                "vel":  vel,
                "ang_vel": ang_vel,
                "rpy": rpy,
                "drpy": drpy,
            }
            
            if self.model_rotor_dynamics: 
                obs["rotor_thrust"] = state_np[..., 12]
        else: 
            obs = {
                "pos": state_np[..., 0:3],
                "quat": state_np[..., 3:7],
                "vel":  state_np[..., 7:10],
                "ang_vel": state_np[..., 10:13],
            }
            
            if self.model_rotor_dynamics: 
                obs["rotor_thrust"] = state_np[..., 13]
        
        return obs
    
    def clip_command(self, cmd:  torch.Tensor) -> torch.Tensor:
        """Clip command to valid ranges.
        
        Args:
            cmd: Command tensor of shape (..., 4)
            
        Returns:
            Clipped command tensor
        """
        cmd_clipped = cmd.clone()
        cmd_clipped[..., 0:3] = torch.clamp(cmd[..., 0:3], -self.rpy_max, self.rpy_max)
        cmd_clipped[..., 3] = torch.clamp(cmd[..., 3], self.thrust_min, self.thrust_max)
        return cmd_clipped
    
    def get_hover_command(self, batch_shape: tuple = ()) -> torch.Tensor:
        """Get hover command (zero RPY, gravity-compensating thrust).
        
        Note: The exact hover thrust depends on acc_coef and cmd_f_coef.
        This returns a nominal value; adjust based on your model calibration.
        
        Args:
            batch_shape: Shape for batched commands
            
        Returns:
            Hover command tensor of shape (*batch_shape, 4)
        """
        # Nominal hover thrust (this is model-dependent)
        hover_thrust = (self.mass * 9.81 - self.acc_coef) / self.cmd_f_coef
        hover_thrust = float(hover_thrust.cpu().numpy())
        
        cmd = torch.zeros((*batch_shape, 4), dtype=self.dtype, device=self.device)
        cmd[..., 3] = hover_thrust
        
        return cmd