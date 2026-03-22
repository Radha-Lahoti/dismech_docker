import numpy as np
import argparse
import py_dismech
from pathlib import Path
from functools import partial


# For thetas if necessary
def parallel_transport(u: np.ndarray, t0: np.ndarray, t1: np.ndarray) -> np.ndarray:
    b = np.cross(t0, t1)
    d = np.dot(t0, t1)
    denom = 1.0 + d + 1e-8
    b_cross_u = np.cross(b, u)
    return u + b_cross_u + np.cross(b, b_cross_u) / denom


def extract_triplet(qs: np.ndarray) -> np.ndarray:
    return np.concatenate([qs[0], [0.0], np.mean(qs, axis=0), [0.0], qs[-1]])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run slinky simulation in DER.")
    parser.add_argument("--radius", type=float, default=5e-3, help="Radius of the rod")
    parser.add_argument("--young_mod", type=float, default=1e7, help="Young's Modulus")
    parser.add_argument("--density", type=float, default=1273.52, help="Density")
    parser.add_argument("--poisson", type=float, default=0.5, help="Poisson ratio")
    parser.add_argument("--dt", type=float, default=5e-3, help="Time step")
    parser.add_argument("--dtol", type=float, default=1e-3, help="Tolerance")
    parser.add_argument("--render", action="store_true", help="Run with OpenGL")
    parser.add_argument("--steps", type=int, default=10, help="Number of saved steps")
    parser.add_argument(
        "--final_disp",
        type=float,
        nargs=3,
        default=[-0.1, 0.0, 0.05],
        help="Final displacement (x y z)",
    )
    args = parser.parse_args()

    # Aliases
    sim_manager = py_dismech.SimulationManager()
    soft_robots = sim_manager.soft_robots
    sim_params = sim_manager.sim_params
    add_force = sim_manager.forces.addForce

    # Enable/disable render
    if args.render:
        sim_manager.render_params.renderer = py_dismech.OPENGL
        sim_manager.render_params.render_scale = 1.0
    else:
        sim_manager.render_params.renderer = py_dismech.HEADLESS

    # Parameters
    final_disp = np.array(args.final_disp)
    sim_params.dt = args.dt
    sim_params.dtol = args.dtol

    # At minimum take dt as step
    steps_per_move = max(int((1.0 / args.steps) / sim_params.dt), 1)
    ddisp = final_disp / (args.steps * steps_per_move)

    # Constants
    sim_params.integrator = py_dismech.BACKWARD_EULER
    vertices = np.loadtxt(Path(__file__).parent / "vertices/slinky_save.txt")
    gravity = np.array([0.0, 0.0, -9.81])
    damping = np.array([2.0])
    velocity_tolerance = 1e-3
    max_settle_steps = 10000

    # Create the helix limb with custom configuration
    add_limb = partial(
        sim_manager.soft_robots.addLimb,
        rho=args.density,
        rod_radius=args.radius,
        youngs_modulus=args.young_mod,
        poisson_ratio=args.poisson,
    )

    # Add the helical structure
    add_limb(vertices)
    helix = sim_manager.soft_robots.limbs[0]

    soft_robots.lockEdge(0, 0)  # first edge
    soft_robots.lockEdge(0, vertices.shape[0] - 2)  # last edge

    add_force(py_dismech.GravityForce(soft_robots, gravity))
    add_force(py_dismech.DampingForce(soft_robots, damping))
    add_force(
        py_dismech.ContactForce(
            soft_robots,
            2.01 * args.radius,
            0.1 * args.radius,
            0.1 * args.young_mod,
            True,
            0.3,
            True,
        )
    )

    sim_manager.initialize([])
    qs = []
    idx_b = [0, 1, 2, 3, 7, 8, 9, 10]
    xb_m = [[0.0, 0.0, 0.0, 0.0, 0.0, *final_disp]]
    xb_c = [0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0]
    raw = []

    def step_until_static() -> None:
        """Progress simulation until |v| < v_tol or iters > max_settle_steps.

        Raises:
            RuntimeError: If iters > max_settle_steps and |v| > v_tol.
        """
        for i in range(max_settle_steps):
            sim_manager.step_simulation()
            if np.linalg.norm(helix.getVelocities()) < velocity_tolerance:
                vertices = helix.getVertices()
                qs.append(extract_triplet(vertices))
                raw.append(vertices)
                break

        if np.linalg.norm(helix.getVelocities()) > velocity_tolerance:
            raise RuntimeError(
                f"Unable to find a static state: v_norm={np.linalg.norm(helix.getVelocities()):.4f}"
            )

    # Wait until static under gravity
    step_until_static()
    for i in range(args.steps):
        # Take small steps for stability under contact
        for _ in range(steps_per_move):
            sim_manager.step_simulation(
                {
                    "delta_position": np.array(
                        [
                            [0, vertices.shape[0] - 1, *ddisp],
                            [0, vertices.shape[0] - 2, *ddisp],
                        ]
                    )
                }
            )
        step_until_static()

    np.savez(
        "/workspace/slinky_output/output.npz",
        raw=np.asarray(raw),
        qs=np.asarray(qs),
        idx_b=idx_b,
        xb_m=xb_m,
        xb_c=xb_c,
    )
