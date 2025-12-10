"""
JAX implementation of the "so_rpy" Euler-angle dynamics suitable for MPPI rollouts.

This is a direct translation of the equations in `symbolic_dynamics_euler` into JAX.
It is pure JAX (jax.numpy) so it can be jitted, vmapped and differentiated.

Notes:
- rpy order is (roll, pitch, yaw). The rotation matrix implementation here
  follows the conventional intrinsic "xyz" rotation order by composing
  R = R_z(yaw) @ R_y(pitch) @ R_x(roll). If you need a different convention
  (e.g. R = R_x @ R_y @ R_z), please adjust `rpy_to_matrix`.
- All inputs are JAX arrays (or convertible). The function is fully JAX-traceable.
- Coefficients (rpy_coef, rpy_rates_coef, cmd_rpy_coef) can be scalars or
  length-3 vectors; broadcasting is used where appropriate.

Example usage:
  import jax
  import jax.numpy as jnp
  from drone_models.so_rpy.jax_euler_dynamics import jax_euler_dynamics

  # single state
  pos = jnp.zeros(3)
  rpy = jnp.zeros(3)
  vel = jnp.zeros(3)
  drpy = jnp.zeros(3)
  cmd = jnp.array([0., 0., 0., 1.])  # rpy_cmd (3), thrust (1)

  params = dict(
      mass=1.0,
      gravity_vec=jnp.array([0., 0., -9.81]),
      acc_coef=0.0,
      cmd_f_coef=1.0,
      rpy_coef=jnp.array([-1., -1., -1.]),
      rpy_rates_coef=jnp.array([-0.5, -0.5, -0.5]),
      cmd_rpy_coef=jnp.array([1.0, 1.0, 1.0]),
  )

  # jit the dynamics
  f = jax.jit(lambda pos, rpy, vel, drpy, cmd: jax_euler_dynamics(
      pos, rpy, vel, drpy, cmd, **params))
  pos_dot, rpy_dot, vel_dot, ddrpy = f(pos, rpy, vel, drpy, cmd)

  # batched (N, ...) states can be handled with jax.vmap
"""
from __future__ import annotations

from typing import Optional, Tuple

import jax
import jax.numpy as jnp


Array = jnp.ndarray


def rpy_to_matrix(rpy: Array) -> Array:
    """Convert roll-pitch-yaw (rpy) to a rotation matrix.

    rpy: (..., 3) array of [roll, pitch, yaw]
    Returns: (..., 3, 3) rotation matrix.

    Convention: build R = R_z(yaw) @ R_y(pitch) @ R_x(roll).
    """
    roll = rpy[..., 0]
    pitch = rpy[..., 1]
    yaw = rpy[..., 2]

    cr = jnp.cos(roll)
    sr = jnp.sin(roll)
    cp = jnp.cos(pitch)
    sp = jnp.sin(pitch)
    cy = jnp.cos(yaw)
    sy = jnp.sin(yaw)

    # Rotation matrices
    Rx = jnp.stack(
        [jnp.stack([jnp.ones_like(cr), jnp.zeros_like(cr), jnp.zeros_like(cr)], -1),
         jnp.stack([jnp.zeros_like(cr), cr, -sr], -1),
         jnp.stack([jnp.zeros_like(cr), sr, cr], -1)],
        axis=-2,
    )  # (...,3,3)

    Ry = jnp.stack(
        [jnp.stack([cp, jnp.zeros_like(cp), sp], -1),
         jnp.stack([jnp.zeros_like(cp), jnp.ones_like(cp), jnp.zeros_like(cp)], -1),
         jnp.stack([-sp, jnp.zeros_like(cp), cp], -1)],
        axis=-2,
    )

    Rz = jnp.stack(
        [jnp.stack([cy, -sy, jnp.zeros_like(cy)], -1),
         jnp.stack([sy, cy, jnp.zeros_like(cy)], -1),
         jnp.stack([jnp.zeros_like(cy), jnp.zeros_like(cy), jnp.ones_like(cy)], -1)],
        axis=-2,
    )

    # Compose: R = Rz @ Ry @ Rx
    Rzy = jnp.matmul(Rz, Ry)
    R = jnp.matmul(Rzy, Rx)
    return R


def jax_euler_dynamics(
    pos: Array,
    rpy: Array,
    vel: Array,
    drpy: Array,
    cmd_rpyt: Array,
    mass: float,
    gravity_vec: Array,
    acc_coef: Array,
    cmd_f_coef: Array,
    rpy_coef: Array,
    rpy_rates_coef: Array,
    cmd_rpy_coef: Array,
    dist_f: Optional[Array] = None,
) -> Tuple[Array, Array, Array, Array]:
    """Compute Euler-form dynamics in JAX.

    Args:
      pos: (..., 3) position
      rpy: (..., 3) roll, pitch, yaw
      vel: (..., 3) linear velocity
      drpy: (..., 3) rpy rates
      cmd_rpyt: (..., 4) [cmd_roll, cmd_pitch, cmd_yaw, cmd_thrust]
      mass: scalar mass
      gravity_vec: (..., 3) gravity vector (e.g. [0,0,-9.81])
      acc_coef: scalar or (...,) additive acceleration bias
      cmd_f_coef: scalar mapping thrust command to accel term
      rpy_coef: scalar or (...,3) coefficient for rpy in ddrpy
      rpy_rates_coef: scalar or (...,3) coefficient for drpy in ddrpy
      cmd_rpy_coef: scalar or (...,3) coefficient for command in ddrpy
      dist_f: optional (...,3) external force in world frame applied at CoM

    Returns:
      pos_dot, rpy_dot, vel_dot, ddrpy  (each with matching leading dims)
    """
    # Ensure arrays are JAX arrays and shapes broadcast naturally
    pos = jnp.asarray(pos)
    rpy = jnp.asarray(rpy)
    vel = jnp.asarray(vel)
    drpy = jnp.asarray(drpy)
    cmd_rpyt = jnp.asarray(cmd_rpyt)
    gravity_vec = jnp.asarray(gravity_vec)

    # Commands
    cmd_rpy = cmd_rpyt[..., :3]
    cmd_thrust = cmd_rpyt[..., -1]

    # Rotation matrix from body to world
    R = rpy_to_matrix(rpy)  # (..., 3, 3)

    # motor/collective forces (scalar -> z component in body frame)
    # forces_motor_vec is (0,0, acc_coef + cmd_f_coef * thrust)
    thrust_acc = acc_coef + cmd_f_coef * cmd_thrust
    forces_motor_vec = jnp.stack([jnp.zeros_like(thrust_acc), jnp.zeros_like(thrust_acc), thrust_acc], axis=-1)

    # Linear dynamics
    pos_dot = vel
    # vel_dot = (R @ forces_motor_vec) / mass + gravity_vec
    # Use matmul with added trailing axes handling
    vel_dot = jnp.einsum("...ij,...j->...i", R, forces_motor_vec) / mass + gravity_vec

    if dist_f is not None:
        vel_dot = vel_dot + jnp.asarray(dist_f) / mass

    # rpy kinematics
    rpy_dot = drpy

    # second-order rpy dynamics
    ddrpy = rpy_coef * rpy + rpy_rates_coef * drpy + cmd_rpy_coef * cmd_rpy

    return pos_dot, rpy_dot, vel_dot, ddrpy


# Helper wrappers for common state formats -------------------------------------------------
def jax_euler_dynamics_statevec(
    state: Array,
    cmd_rpyt: Array,
    mass: float,
    gravity_vec: Array,
    acc_coef: Array,
    cmd_f_coef: Array,
    rpy_coef: Array,
    rpy_rates_coef: Array,
    cmd_rpy_coef: Array,
    dist_f: Optional[Array] = None,
) -> Array:
    """State-vector-style wrapper.

    State layout: (..., 12) = [pos(3), rpy(3), vel(3), drpy(3)]
    Returns state derivative in same layout.
    """
    pos = state[..., 0:3]
    rpy = state[..., 3:6]
    vel = state[..., 6:9]
    drpy = state[..., 9:12]

    pos_dot, rpy_dot, vel_dot, ddrpy = jax_euler_dynamics(
        pos,
        rpy,
        vel,
        drpy,
        cmd_rpyt,
        mass=mass,
        gravity_vec=gravity_vec,
        acc_coef=acc_coef,
        cmd_f_coef=cmd_f_coef,
        rpy_coef=rpy_coef,
        rpy_rates_coef=rpy_rates_coef,
        cmd_rpy_coef=cmd_rpy_coef,
        dist_f=dist_f,
    )
    return jnp.concatenate([pos_dot, rpy_dot, vel_dot, ddrpy], axis=-1)


# Vectorized version for batch rollouts (along leading axis)
def vmap_jax_euler_dynamics_statevec(
    states: Array,
    cmds: Array,
    mass: float,
    gravity_vec: Array,
    acc_coef: Array,
    cmd_f_coef: Array,
    rpy_coef: Array,
    rpy_rates_coef: Array,
    cmd_rpy_coef: Array,
    dists_f: Optional[Array] = None,
):
    """Vectorized dynamics over a leading batch dimension using jax.vmap.

    states: (N, 12) or (N, ..., 12)
    cmds: (N, 4)
    dists_f: optional (N, 3)
    """
    fn = lambda s, u, df: jax_euler_dynamics_statevec(
        s,
        u,
        mass,
        gravity_vec,
        acc_coef,
        cmd_f_coef,
        rpy_coef,
        rpy_rates_coef,
        cmd_rpy_coef,
        dist_f=df,
    )
    if dists_f is None:
        # map with None -> use a dummy array of zeros
        dists_none = jnp.zeros(states.shape[:-1] + (3,))
        return jax.vmap(fn)(states, cmds, dists_none)
    else:
        return jax.vmap(fn)(states, cmds, dists_f)