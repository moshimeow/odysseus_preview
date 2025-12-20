//! 2D Robot Arm Inverse Kinematics
//!
//! Matches the original arena_autodiff example - fitting joint angles to match
//! BOTH joint1 position AND end effector position.

use nalgebra::{SMatrix, SVector};
use odysseus_solver::{Jet, LevenbergMarquardt, Real};
use std::f64::consts::PI;

#[cfg(feature = "visualization")]
use rerun as rr;

const N_JOINTS: usize = 2;
const N_RESIDUALS: usize = 4; // joint1_x, joint1_y, end_x, end_y

type Jet2 = Jet<f64, N_JOINTS>;
type Params = SVector<f64, N_JOINTS>;
type Residuals = SVector<f64, N_RESIDUALS>;
type Jacobian = SMatrix<f64, N_RESIDUALS, N_JOINTS>;

#[derive(Debug, Clone, Copy)]
struct Point2D {
    x: f64,
    y: f64,
}

impl Point2D {
    fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(feature = "visualization")]
    let rec = rr::RecordingStreamBuilder::new("robot_arm_2d").spawn()?;

    println!("🦾 odysseus-solver: 2D Robot Arm IK");
    println!("====================================\n");

    // Problem setup
    let l1 = 2.0;
    let l2 = 2.0;

    // Target configuration
    let true_theta1 = PI;           // 180°
    let true_theta2 = PI / 6.0;     // 30°
    let (target_joint1, target_end) = forward_kinematics_f64(true_theta1, true_theta2, l1, l2);

    println!("🎯 Target Configuration:");
    println!("  True angles: θ₁={:.1}°, θ₂={:.1}°",
        true_theta1.to_degrees(), true_theta2.to_degrees());
    println!("  Target joint1: ({:.3}, {:.3})", target_joint1.x, target_joint1.y);
    println!("  Target end:    ({:.3}, {:.3})", target_end.x, target_end.y);

    // Initial guess (straight right)
    let mut params = Params::new(0.0, 0.0);
    println!("\n🚀 Initial guess: θ₁=0°, θ₂=0°\n");

    // Cost function
    let cost_fn = |params: &Params| {
        compute_residuals_and_jacobian(params, l1, l2, target_joint1, target_end)
    };

    // Solve
    let mut solver = LevenbergMarquardt::<f64, N_JOINTS, N_RESIDUALS>::new()
        .with_tolerance(1e-10)
        .with_max_iterations(30)
        .with_verbose(false);

    println!("🔧 Starting Optimization:");

    // Wrap the cost function to make it compatible with solve()
    let cost_fn_wrapped = |params: &Params,
                           residuals: &mut Residuals,
                           jacobian: &mut Jacobian| {
        let (r, j) = cost_fn(params);
        *residuals = r;
        *jacobian = j;
    };

    params = solver.solve(params, cost_fn_wrapped, |iter, result, params| {
        if iter < 5 || iter % 5 == 0 {
            println!("  Iter {:2}: error={:.6}, θ=[{:.3}, {:.3}] ({:.1}°, {:.1}°)",
                iter, result.error, params[0], params[1],
                params[0].to_degrees(), params[1].to_degrees());
        }

        #[cfg(feature = "visualization")]
        {
            rec.set_time_sequence("iteration", iter as i64);
            let (current_joint1, current_end) = forward_kinematics_f64(params[0], params[1], l1, l2);
            log_robot_arms(&rec, current_joint1, current_end, target_joint1, target_end);
        }
    });

    // Results
    let (final_joint1, final_end) = forward_kinematics_f64(params[0], params[1], l1, l2);

    println!("\n🎉 Solution Found:");
    println!("  Estimated angles: θ₁={:.3} rad ({:.1}°)", params[0], params[0].to_degrees());
    println!("                    θ₂={:.3} rad ({:.1}°)", params[1], params[1].to_degrees());
    println!("  True angles:      θ₁={:.3} rad ({:.1}°)", true_theta1, true_theta1.to_degrees());
    println!("                    θ₂={:.3} rad ({:.1}°)", true_theta2, true_theta2.to_degrees());

    let error_joint1 = ((final_joint1.x - target_joint1.x).powi(2) +
                        (final_joint1.y - target_joint1.y).powi(2)).sqrt();
    let error_end = ((final_end.x - target_end.x).powi(2) +
                     (final_end.y - target_end.y).powi(2)).sqrt();

    println!("\n📊 Position Errors:");
    println!("  Joint1: {:.2e}", error_joint1);
    println!("  End:    {:.2e}", error_end);

    #[cfg(feature = "visualization")]
    {
        rec.set_time_sequence("iteration", 100);
        log_robot_arms(&rec, final_joint1, final_end, target_joint1, target_end);
        println!("\n📺 Open Rerun to see the optimization!");
        println!("   Gray = Target configuration");
        println!("   Blue = Current estimate");
    }

    Ok(())
}

/// Forward kinematics - returns (joint1, end_effector)
fn forward_kinematics_f64(theta1: f64, theta2: f64, l1: f64, l2: f64) -> (Point2D, Point2D) {
    let joint1 = Point2D::new(l1 * theta1.cos(), l1 * theta1.sin());
    let absolute_theta2 = theta1 + theta2;
    let end = Point2D::new(
        joint1.x + l2 * absolute_theta2.cos(),
        joint1.y + l2 * absolute_theta2.sin(),
    );
    (joint1, end)
}

/// Generic forward kinematics with Jets
fn forward_kinematics_jet<T: Real>(theta1: T, theta2: T, l1: f64, l2: f64) -> (Point2D_<T>, Point2D_<T>) {
    let l1_t = T::from_f64(l1);
    let l2_t = T::from_f64(l2);

    let joint1_x = l1_t * theta1.cos();
    let joint1_y = l1_t * theta1.sin();

    let absolute_theta2 = theta1 + theta2;
    let end_x = joint1_x + l2_t * absolute_theta2.cos();
    let end_y = joint1_y + l2_t * absolute_theta2.sin();

    (Point2D_ { x: joint1_x, y: joint1_y }, Point2D_ { x: end_x, y: end_y })
}

#[derive(Debug, Clone, Copy)]
struct Point2D_<T> {
    x: T,
    y: T,
}

fn compute_residuals_and_jacobian(
    params: &Params,
    l1: f64,
    l2: f64,
    target_joint1: Point2D,
    target_end: Point2D,
) -> (Residuals, Jacobian) {
    let theta1 = Jet2::variable(params[0], 0);
    let theta2 = Jet2::variable(params[1], 1);

    let (joint1, end) = forward_kinematics_jet(theta1, theta2, l1, l2);

    // Residuals: current - target
    let r_j1x = joint1.x - Jet2::constant(target_joint1.x);
    let r_j1y = joint1.y - Jet2::constant(target_joint1.y);
    let r_ex = end.x - Jet2::constant(target_end.x);
    let r_ey = end.y - Jet2::constant(target_end.y);

    let mut residuals = Residuals::zeros();
    let mut jacobian = Jacobian::zeros();

    residuals[0] = r_j1x.value;
    residuals[1] = r_j1y.value;
    residuals[2] = r_ex.value;
    residuals[3] = r_ey.value;

    for j in 0..N_JOINTS {
        jacobian[(0, j)] = r_j1x.derivs[j];
        jacobian[(1, j)] = r_j1y.derivs[j];
        jacobian[(2, j)] = r_ex.derivs[j];
        jacobian[(3, j)] = r_ey.derivs[j];
    }

    (residuals, jacobian)
}

#[cfg(feature = "visualization")]
fn log_robot_arms(
    rec: &rr::RecordingStream,
    current_joint1: Point2D,
    current_end: Point2D,
    target_joint1: Point2D,
    target_end: Point2D,
) {
    // Target arm (gray)
    rec.log(
        "world/robot_arms/target/link1",
        &rr::LineStrips2D::new([[[0.0, 0.0], [target_joint1.x as f32, target_joint1.y as f32]]])
            .with_colors([[150, 150, 150]]),
    ).unwrap();

    rec.log(
        "world/robot_arms/target/link2",
        &rr::LineStrips2D::new([[[target_joint1.x as f32, target_joint1.y as f32],
                                   [target_end.x as f32, target_end.y as f32]]])
            .with_colors([[150, 150, 150]]),
    ).unwrap();

    rec.log(
        "world/robot_arms/target/joints",
        &rr::Points2D::new([[0.0, 0.0],
                             [target_joint1.x as f32, target_joint1.y as f32],
                             [target_end.x as f32, target_end.y as f32]])
            .with_colors([[150, 150, 150]])
            .with_radii([0.1, 0.08, 0.12]),
    ).unwrap();

    // Current arm (blue)
    rec.log(
        "world/robot_arms/current/link1",
        &rr::LineStrips2D::new([[[0.0, 0.0], [current_joint1.x as f32, current_joint1.y as f32]]])
            .with_colors([[100, 100, 255]]),
    ).unwrap();

    rec.log(
        "world/robot_arms/current/link2",
        &rr::LineStrips2D::new([[[current_joint1.x as f32, current_joint1.y as f32],
                                   [current_end.x as f32, current_end.y as f32]]])
            .with_colors([[100, 100, 255]]),
    ).unwrap();

    rec.log(
        "world/robot_arms/current/joints",
        &rr::Points2D::new([[0.0, 0.0],
                             [current_joint1.x as f32, current_joint1.y as f32],
                             [current_end.x as f32, current_end.y as f32]])
            .with_colors([[100, 100, 255]])
            .with_radii([0.1, 0.08, 0.12]),
    ).unwrap();
}
