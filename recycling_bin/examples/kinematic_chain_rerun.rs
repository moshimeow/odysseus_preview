use rerun::{self as rr, RecordingStream};
use arena_autodiff::{expr, TinySolver, TinySolverBuffers, ResidualWriter, CostFunctor};
use std::f64::consts::PI;

/// 2D point for visualization
#[derive(Debug, Clone, Copy)]
struct Point2D {
    x: f64,
    y: f64,
}

impl Point2D {
    fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    fn distance_to(&self, other: &Point2D) -> f64 {
        ((self.x - other.x).powi(2) + (self.y - other.y).powi(2)).sqrt()
    }
}

/// Standard forward kinematics for visualization
fn forward_kinematics(theta1: f64, theta2: f64, l1: f64, l2: f64) -> (Point2D, Point2D) {
    // First joint position
    let joint1 = Point2D::new(l1 * theta1.cos(), l1 * theta1.sin());

    // End of second link (θ2 is relative to first link)
    let absolute_theta2 = theta1 + theta2;
    let end_effector = Point2D::new(
        joint1.x + l2 * absolute_theta2.cos(),
        joint1.y + l2 * absolute_theta2.sin(),
    );

    (joint1, end_effector)
}


/// Log robot arm visualization to Rerun
fn log_robot_arm(rec: &RecordingStream, label: &str, joint1: Point2D, end_effector: Point2D, color: [u8; 3]) {
    // Base to joint1
    rec.log(
        format!("world/robot_arms/{}/link1", label),
        &rr::LineStrips2D::new([[[0.0, 0.0], [joint1.x as f32, joint1.y as f32]]])
            .with_colors([color]),
    ).unwrap();

    // Joint1 to end effector
    rec.log(
        format!("world/robot_arms/{}/link2", label),
        &rr::LineStrips2D::new([[[joint1.x as f32, joint1.y as f32], [end_effector.x as f32, end_effector.y as f32]]])
            .with_colors([color]),
    ).unwrap();

    // Joints as circles (scaled appropriately for 2-unit link lengths)
    rec.log(
        format!("world/robot_arms/{}/base", label),
        &rr::Points2D::new([[0.0, 0.0]])
            .with_colors([color])
            .with_radii([0.1]),
    ).unwrap();

    rec.log(
        format!("world/robot_arms/{}/joint1", label),
        &rr::Points2D::new([[joint1.x as f32, joint1.y as f32]])
            .with_colors([color])
            .with_radii([0.08]),
    ).unwrap();

    rec.log(
        format!("world/robot_arms/{}/end_effector", label),
        &rr::Points2D::new([[end_effector.x as f32, end_effector.y as f32]])
            .with_colors([color])
            .with_radii([0.12]),
    ).unwrap();
}

/// Inverse kinematics cost functor for a 2-link robot arm
struct InverseKinematicsCostFunctor {
    l1: f64,
    l2: f64,
    target_joint1: Point2D,
    target_end: Point2D,
}

impl InverseKinematicsCostFunctor {
    fn new(l1: f64, l2: f64, target_joint1: Point2D, target_end: Point2D) -> Self {
        Self { l1, l2, target_joint1, target_end }
    }
}

// Single generic implementation that works for ANY MathContext!
// This works for JetArena<f64> (with Jacobian), F64Ctx (values only), F32Ctx, etc.
impl<Ctx: arena_autodiff::MathContext> CostFunctor<Ctx> for InverseKinematicsCostFunctor {
    fn residuals(&self, params: &[Ctx::Value], ctx: &mut Ctx, mut writer: ResidualWriter<Ctx::Value>) -> Result<usize, String> {
        // Extract parameters (works for both JetHandle and f64!)
        let theta1 = params[0];
        let theta2 = params[1];

        // Convert constants to context values
        let l1 = ctx.constant(self.l1);
        let l2 = ctx.constant(self.l2);
        let target_j1x = ctx.constant(self.target_joint1.x);
        let target_j1y = ctx.constant(self.target_joint1.y);
        let target_ex = ctx.constant(self.target_end.x);
        let target_ey = ctx.constant(self.target_end.y);

        // Compute forward kinematics: (L1*cos(θ1), L1*sin(θ1))
        let joint1_x = expr!(ctx, l1 * cos(theta1));
        let joint1_y = expr!(ctx, l1 * sin(theta1));

        // End effector: joint1 + L2*(cos(θ1+θ2), sin(θ1+θ2))
        let end_x = expr!(ctx, joint1_x + l2 * cos(theta1 + theta2));
        let end_y = expr!(ctx, joint1_y + l2 * sin(theta1 + theta2));

        // Compute residuals: current - target
        let residual_j1x = expr!(ctx, joint1_x - target_j1x);
        let residual_j1y = expr!(ctx, joint1_y - target_j1y);
        let residual_ex = expr!(ctx, end_x - target_ex);
        let residual_ey = expr!(ctx, end_y - target_ey);

        // Write residuals using safe ResidualWriter
        writer.push(residual_j1x).map_err(|e| format!("Buffer error: {}", e))?;
        writer.push(residual_j1y).map_err(|e| format!("Buffer error: {}", e))?;
        writer.push(residual_ex).map_err(|e| format!("Buffer error: {}", e))?;
        writer.push(residual_ey).map_err(|e| format!("Buffer error: {}", e))?;

        Ok(writer.finish())
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Connect to your existing Rerun viewer
    let rec = rr::RecordingStreamBuilder::new("arena_autodiff_kinematic_chain")
        .spawn()?;

    // Problem setup
    let l1 = 2.0; // Length of first link
    let l2 = 2.0; // Length of second link

    // Target configuration (what we want to reach)
    let true_theta1 = PI; // 180°
    let true_theta2 = PI / 6.0; // 30°
    let (target_joint1, target_end) = forward_kinematics(true_theta1, true_theta2, l1, l2);

    println!("🦾 arena_autodiff 2D Kinematic Chain - Inverse Kinematics with Rerun Visualization");
    println!("===============================================================================");
    println!("🎯 Target Configuration:");
    println!("True angles: θ₁={:.1}°, θ₂={:.1}°", true_theta1 * 180.0 / PI, true_theta2 * 180.0 / PI);
    println!("Target joint1: ({:.3}, {:.3})", target_joint1.x, target_joint1.y);
    println!("Target end effector: ({:.3}, {:.3})", target_end.x, target_end.y);

    // Create inverse kinematics cost functor using clean API
    // 🚀 ZERO-ALLOCATION COST FUNCTION with CostFunctor trait!
    let cost_functor = InverseKinematicsCostFunctor::new(l1, l2, target_joint1, target_end);

    // Starting configuration (far from solution)
    let initial_guess = [PI / 6.0, -PI / 4.0]; // 30°, -45°

    println!("\n🚀 Starting Configuration:");
    let (initial_joint1, initial_end) = forward_kinematics(initial_guess[0], initial_guess[1], l1, l2);
    println!("Initial angles: θ₁={:.1}°, θ₂={:.1}°", initial_guess[0] * 180.0 / PI, initial_guess[1] * 180.0 / PI);
    println!("Initial joint1: ({:.3}, {:.3})", initial_joint1.x, initial_joint1.y);
    println!("Initial end effector: ({:.3}, {:.3})", initial_end.x, initial_end.y);

    // 🚀 ZERO-ALLOCATION SETUP: Pre-allocate all buffers!
    let num_params = 2; // θ₁, θ₂
    let max_residuals = 4; // joint1_x, joint1_y, end_x, end_y
    let mut buffers = TinySolverBuffers::new(num_params, max_residuals);

    // Solve using Levenberg-Marquardt with zero-allocation API
    let solver = TinySolver::new()
        .with_tolerance(1e-10)
        .with_max_iterations(20)
        .with_levenberg_marquardt(true);

    // 🚀 ITERATION-BY-ITERATION OPTIMIZATION with timeline logging!
    let mut params = initial_guess.to_vec();
    let mut lambda = 1e-3; // Initial LM damping parameter

    println!("\n🚀 Starting Levenberg-Marquardt Optimization:");

    for iteration in 0..20 { // max_iterations
        // Log current state to timeline
        rec.set_time_sequence("optimization_step", iteration as i64);

        // Log target on every timeline step so it's always visible
        log_robot_arm(&rec, "target", target_joint1, target_end, [128, 128, 128]);

        let (current_joint1, current_end) = forward_kinematics(params[0], params[1], l1, l2);
        log_robot_arm(&rec, "current", current_joint1, current_end, [255, 150, 50]);

        // Compute current errors for display
        let joint1_error = current_joint1.distance_to(&target_joint1);
        let end_error = current_end.distance_to(&target_end);
        let total_error = (joint1_error * joint1_error + end_error * end_error).sqrt();

        println!("Iteration {}: θ₁={:.3}°, θ₂={:.3}° | Joint1_err:{:.4}, End_err:{:.4}, Total:{:.6}",
                 iteration, params[0] * 180.0 / PI, params[1] * 180.0 / PI,
                 joint1_error, end_error, total_error);

        // Perform one optimization step (LM with line search!)
        match solver.solve_iteration_with_functor(&mut params, &mut buffers, &cost_functor, &mut lambda) {
            Ok((step_norm, converged)) => {
                if converged {
                    println!("✅ Converged after {} iterations (step norm: {:.2e}, λ: {:.2e})", iteration + 1, step_norm, lambda);
                    break;
                }
            }
            Err(e) => {
                println!("❌ Optimization failed at iteration {}: {}", iteration, e);
                break;
            }
        }
    }

    // Log final solution
    let (final_joint1, final_end) = forward_kinematics(params[0], params[1], l1, l2);
    rec.set_time_sequence("optimization_step", 20); // max_iterations
    log_robot_arm(&rec, "target", target_joint1, target_end, [128, 128, 128]);
    log_robot_arm(&rec, "current", final_joint1, final_end, [100, 255, 100]);

    println!("\n🎉 Final Results:");
    println!("Final angles: θ₁={:.6}° ({:.6} rad), θ₂={:.6}° ({:.6} rad)",
             params[0] * 180.0 / PI, params[0], params[1] * 180.0 / PI, params[1]);
    println!("True angles:  θ₁={:.6}° ({:.6} rad), θ₂={:.6}° ({:.6} rad)",
             true_theta1 * 180.0 / PI, true_theta1, true_theta2 * 180.0 / PI, true_theta2);

    let angle_error1 = (params[0] - true_theta1).abs();
    let angle_error2 = (params[1] - true_theta2).abs().min((params[1] - true_theta2 + 2.0 * PI).abs());
    println!("Angle errors: Δθ₁={:.2e} rad ({:.4}°), Δθ₂={:.2e} rad ({:.4}°)",
             angle_error1, angle_error1 * 180.0 / PI,
             angle_error2, angle_error2 * 180.0 / PI);

    println!("\n🎬 Open the Rerun viewer to see the step-by-step optimization!");
    println!("   - Target robot arm is shown in gray (static)");
    println!("   - Current configuration animates through the timeline");
    println!("   - Use the timeline slider to see each iteration!");

    Ok(())
}