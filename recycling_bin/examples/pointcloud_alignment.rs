use rerun::{self as rr, RecordingStream};
use arena_autodiff::{
    expr, math3d, CostFunctor, F64Ctx, MathContext, ResidualWriter, TinySolver,
    TinySolverBuffers,
};

/// Generate random points in a [-1, 1]³ cube
fn generate_pointcloud(num_points: usize, seed: u64) -> Vec<math3d::Vec3<f64>> {
    // Simple LCG random number generator for reproducibility
    let mut rng = seed;
    let lcg = |r: &mut u64| -> f64 {
        *r = r.wrapping_mul(1103515245).wrapping_add(12345);
        ((*r / 65536) % 32768) as f64 / 32768.0 * 2.0 - 1.0
    };

    (0..num_points)
        .map(|_| math3d::Vec3::new(lcg(&mut rng), lcg(&mut rng), lcg(&mut rng)))
        .collect()
}

/// Check if a rotation matrix is orthonormal
fn check_rotation_matrix_quality(rotation: &math3d::Mat3<f64>) -> (f64, f64, f64) {
    // Check column lengths (should be 1.0)
    let col0_len = (rotation.x_axis.x * rotation.x_axis.x
        + rotation.x_axis.y * rotation.x_axis.y
        + rotation.x_axis.z * rotation.x_axis.z)
        .sqrt();
    let col1_len = (rotation.y_axis.x * rotation.y_axis.x
        + rotation.y_axis.y * rotation.y_axis.y
        + rotation.y_axis.z * rotation.y_axis.z)
        .sqrt();
    let col2_len = (rotation.z_axis.x * rotation.z_axis.x
        + rotation.z_axis.y * rotation.z_axis.y
        + rotation.z_axis.z * rotation.z_axis.z)
        .sqrt();

    // Check orthogonality (dot products should be 0)
    let dot_01 = rotation.x_axis.x * rotation.y_axis.x
        + rotation.x_axis.y * rotation.y_axis.y
        + rotation.x_axis.z * rotation.y_axis.z;
    let dot_02 = rotation.x_axis.x * rotation.z_axis.x
        + rotation.x_axis.y * rotation.z_axis.y
        + rotation.x_axis.z * rotation.z_axis.z;
    let dot_12 = rotation.y_axis.x * rotation.z_axis.x
        + rotation.y_axis.y * rotation.z_axis.y
        + rotation.y_axis.z * rotation.z_axis.z;

    let max_col_error = (col0_len - 1.0)
        .abs()
        .max((col1_len - 1.0).abs())
        .max((col2_len - 1.0).abs());
    let max_ortho_error = dot_01.abs().max(dot_02.abs()).max(dot_12.abs());

    // Determinant (should be 1.0 for proper rotation)
    let det = rotation.x_axis.x
        * (rotation.y_axis.y * rotation.z_axis.z - rotation.y_axis.z * rotation.z_axis.y)
        - rotation.x_axis.y
            * (rotation.y_axis.x * rotation.z_axis.z - rotation.y_axis.z * rotation.z_axis.x)
        + rotation.x_axis.z
            * (rotation.y_axis.x * rotation.z_axis.y - rotation.y_axis.y * rotation.z_axis.x);

    (max_col_error, max_ortho_error, (det - 1.0).abs())
}

/// Transform a pointcloud using a pose (translation + rodrigues rotation)
/// Generic over any MathContext for use in both evaluation and autodiff
fn transform_pointcloud<Ctx: MathContext>(
    ctx: &mut Ctx,
    points: &[math3d::Vec3<f64>],
    translation: math3d::Vec3<Ctx::Value>,
    rvec: math3d::Vec3<Ctx::Value>,
) -> Vec<math3d::Vec3<Ctx::Value>> {
    // Compute rotation matrix from rodrigues vector (once for all points)
    let rotation = math3d::rodrigues_to_matrix(ctx, rvec);

    // Transform each point
    points
        .iter()
        .map(|&p| {
            // Convert source point to context values
            let source_pt = math3d::Vec3::new(
                ctx.constant(p.x),
                ctx.constant(p.y),
                ctx.constant(p.z),
            );
            // Apply transformation
            math3d::transform_point(ctx, rotation, translation, source_pt)
        })
        .collect()
}

/// Transform a pointcloud with f64 and check rotation matrix quality
fn transform_pointcloud_f64(
    points: &[math3d::Vec3<f64>],
    translation: math3d::Vec3<f64>,
    rvec: math3d::Vec3<f64>,
) -> Vec<math3d::Vec3<f64>> {
    let mut ctx = F64Ctx;

    // Compute rotation for quality check
    let rotation = math3d::rodrigues_to_matrix(&mut ctx, rvec);
    let (col_err, ortho_err, det_err) = check_rotation_matrix_quality(&rotation);
    println!(
        "  Rotation matrix quality: col_err={:.2e}, ortho_err={:.2e}, det_err={:.2e}",
        col_err, ortho_err, det_err
    );
    if col_err > 0.01 || ortho_err > 0.01 || det_err > 0.01 {
        println!("  ⚠️  WARNING: Rotation matrix is not orthonormal!");
    }

    // Use generic transform function
    transform_pointcloud(&mut ctx, points, translation, rvec)
}

/// Cost functor for pointcloud alignment
///
/// Residuals: difference between current transformed source points and target points
struct PointcloudAlignmentCostFunctor {
    source_points: Vec<math3d::Vec3<f64>>,
    target_points: Vec<math3d::Vec3<f64>>,
}

impl PointcloudAlignmentCostFunctor {
    fn new(source_points: Vec<math3d::Vec3<f64>>, target_points: Vec<math3d::Vec3<f64>>) -> Self {
        assert_eq!(source_points.len(), target_points.len());
        Self {
            source_points,
            target_points,
        }
    }
}

// Generic implementation over any MathContext!
impl<Ctx: MathContext> CostFunctor<Ctx> for PointcloudAlignmentCostFunctor {
    fn residuals(
        &self,
        params: &[Ctx::Value],
        ctx: &mut Ctx,
        mut writer: ResidualWriter<Ctx::Value>,
    ) -> Result<usize, String> {
        // Parameters: [tx, ty, tz, rx, ry, rz]
        // Translation and rodrigues rotation vector
        let translation = math3d::Vec3::new(params[0], params[1], params[2]);
        let rvec = math3d::Vec3::new(params[3], params[4], params[5]);

        // Transform all source points using the generic function
        let transformed_points = transform_pointcloud(ctx, &self.source_points, translation, rvec);

        // Compute residuals: transformed_source - target
        for (transformed, target) in transformed_points.iter().zip(self.target_points.iter()) {
            // Convert target to context values
            let target_pt = math3d::Vec3::new(
                ctx.constant(target.x),
                ctx.constant(target.y),
                ctx.constant(target.z),
            );

            // Compute residuals (x, y, z differences)
            let residual_x = expr!(ctx, transformed.x - target_pt.x);
            let residual_y = expr!(ctx, transformed.y - target_pt.y);
            let residual_z = expr!(ctx, transformed.z - target_pt.z);

            writer
                .push(residual_x)
                .map_err(|e| format!("Buffer error: {}", e))?;
            writer
                .push(residual_y)
                .map_err(|e| format!("Buffer error: {}", e))?;
            writer
                .push(residual_z)
                .map_err(|e| format!("Buffer error: {}", e))?;
        }

        Ok(writer.finish())
    }
}

/// Log pointcloud to Rerun
fn log_pointcloud(
    rec: &RecordingStream,
    entity_path: &str,
    points: &[math3d::Vec3<f64>],
    color: [u8; 3],
    radius: f32,
) {
    let positions: Vec<[f32; 3]> = points
        .iter()
        .map(|p| [p.x as f32, p.y as f32, p.z as f32])
        .collect();

    rec.log(
        entity_path,
        &rr::Points3D::new(positions)
            .with_colors([color])
            .with_radii([radius]),
    )
    .unwrap();
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Connect to Rerun
    let rec = rr::RecordingStreamBuilder::new("arena_autodiff pointcloud_alignment").spawn()?;

    println!("🎯 arena_autodiff 3D Pointcloud Alignment - Pose Estimation");
    println!("====================================================");

    // Generate source pointcloud
    let num_points = 60;
    let source_points = generate_pointcloud(num_points, 67);
    println!("Generated {} source points", num_points);

    // Define target pose (ground truth)
    let true_translation = math3d::Vec3::new(0.3, 0.2, 0.1);
    let true_rvec = math3d::Vec3::new(2.2, 0.3, 1.5); // Small rotation for testing
    println!("\n🎯 True Pose:");
    println!(
        "  Translation: [{:.3}, {:.3}, {:.3}]",
        true_translation.x, true_translation.y, true_translation.z
    );
    println!(
        "  Rotation (rvec): [{:.3}, {:.3}, {:.3}]",
        true_rvec.x, true_rvec.y, true_rvec.z
    );

    // Generate target pointcloud by transforming source
    let target_points = transform_pointcloud_f64(&source_points, true_translation, true_rvec);

    // Initial guess (identity transform)
    let initial_translation = math3d::Vec3::new(0.0, 0.0, 0.0);
    let initial_rvec = math3d::Vec3::new(0.0, 0.0, 0.0);
    let mut params = vec![
        initial_translation.x,
        initial_translation.y,
        initial_translation.z,
        initial_rvec.x,
        initial_rvec.y,
        initial_rvec.z,
    ];

    println!("\n🚀 Initial Guess:");
    println!(
        "  Translation: [{:.3}, {:.3}, {:.3}]",
        params[0], params[1], params[2]
    );
    println!(
        "  Rotation (rvec): [{:.3}, {:.3}, {:.3}]",
        params[3], params[4], params[5]
    );

    // Set up cost functor and solver
    let cost_functor =
        PointcloudAlignmentCostFunctor::new(source_points.clone(), target_points.clone());

    let num_params = 6; // tx, ty, tz, rx, ry, rz
    let max_residuals = num_points * 3; // 3 residuals per point
    let mut buffers = TinySolverBuffers::new(num_params, max_residuals);

    let solver = TinySolver::new()
        .with_tolerance(1e-10)
        .with_max_iterations(30)
        .with_levenberg_marquardt(true)
        .with_arena_capacity(2048);  // Large capacity for complex Rodrigues derivatives

    // Optimization loop with Rerun logging
    let mut lambda = 1e-3;
    println!("\n🔧 Starting Levenberg-Marquardt Optimization:");

    for iteration in 0..30 {
        // Set timeline
        rec.set_time_sequence("optimization_step", iteration as i64);

        // Log current state
        let current_translation = math3d::Vec3::new(params[0], params[1], params[2]);
        let current_rvec = math3d::Vec3::new(params[3], params[4], params[5]);
        let current_transformed =
            transform_pointcloud_f64(&source_points, current_translation, current_rvec);

        // Log pointclouds
        log_pointcloud(&rec, "world/source", &source_points, [100, 100, 255], 0.03); // Blue
        log_pointcloud(&rec, "world/target", &target_points, [150, 150, 150], 0.02); // Gray
        log_pointcloud(
            &rec,
            "world/current",
            &current_transformed,
            [255, 150, 50],
            0.025,
        ); // Orange

        // Compute error for display
        let error: f64 = current_transformed
            .iter()
            .zip(target_points.iter())
            .map(|(c, t)| {
                let dx = c.x - t.x;
                let dy = c.y - t.y;
                let dz = c.z - t.z;
                dx * dx + dy * dy + dz * dz
            })
            .sum::<f64>()
            .sqrt();

        println!(
            "  Iter {:2}: error={:.6}, λ={:.2e}",
            iteration, error, lambda
        );

        // Optimization step
        match solver.solve_iteration_with_functor(&mut params, &mut buffers, &cost_functor, &mut lambda)
        {
            Ok((step_norm, converged)) => {
                if iteration < 2 {
                    println!("    step=[{:.3e}, {:.3e}, {:.3e}, {:.3e}, {:.3e}, {:.3e}], norm={:.3e}",
                             buffers.param_step[0], buffers.param_step[1], buffers.param_step[2],
                             buffers.param_step[3], buffers.param_step[4], buffers.param_step[5], step_norm);
                    println!("    J[0:3,0]=[{:.3e}, {:.3e}, {:.3e}] (∂residual/∂tx)",
                             buffers.jacobian[0], buffers.jacobian[1], buffers.jacobian[2]);
                    println!("    J[0:3,3]=[{:.3e}, {:.3e}, {:.3e}] (∂residual/∂rx)",
                             buffers.jacobian[3], buffers.jacobian[4], buffers.jacobian[5]);
                }
                if converged {
                    println!(
                        "✅ Converged after {} iterations (step norm: {:.2e})",
                        iteration + 1,
                        step_norm
                    );
                    break;
                }
            }
            Err(e) => {
                println!("❌ Optimization failed at iteration {}: {}", iteration, e);
                break;
            }
        }
    }

    // Log final result
    rec.set_time_sequence("optimization_step", 30);
    let final_translation = math3d::Vec3::new(params[0], params[1], params[2]);
    let final_rvec = math3d::Vec3::new(params[3], params[4], params[5]);
    let final_transformed = transform_pointcloud_f64(&source_points, final_translation, final_rvec);

    log_pointcloud(&rec, "world/source", &source_points, [100, 100, 255], 0.03);
    log_pointcloud(&rec, "world/target", &target_points, [150, 150, 150], 0.02);
    log_pointcloud(
        &rec,
        "world/current",
        &final_transformed,
        [100, 255, 100],
        0.025,
    );

    println!("\n🎉 Final Results:");
    println!(
        "  Estimated Translation: [{:.6}, {:.6}, {:.6}]",
        params[0], params[1], params[2]
    );
    println!(
        "  True Translation:      [{:.6}, {:.6}, {:.6}]",
        true_translation.x, true_translation.y, true_translation.z
    );
    println!(
        "  Estimated Rotation:    [{:.6}, {:.6}, {:.6}]",
        params[3], params[4], params[5]
    );
    println!(
        "  True Rotation:         [{:.6}, {:.6}, {:.6}]",
        true_rvec.x, true_rvec.y, true_rvec.z
    );

    let translation_error = (
        (params[0] - true_translation.x).powi(2)
            + (params[1] - true_translation.y).powi(2)
            + (params[2] - true_translation.z).powi(2)
    )
    .sqrt();

    let rotation_error = (
        (params[3] - true_rvec.x).powi(2)
            + (params[4] - true_rvec.y).powi(2)
            + (params[5] - true_rvec.z).powi(2)
    )
    .sqrt();

    println!("\n📊 Errors:");
    println!("  Translation error: {:.2e}", translation_error);
    println!("  Rotation error:    {:.2e}", rotation_error);

    println!("\n🎬 Open the Rerun viewer to see the optimization!");
    println!("   - Source points: Blue");
    println!("   - Target points: Gray");
    println!("   - Current estimate: Orange → Green");
    println!("   - Use the timeline slider to see each iteration!");

    Ok(())
}
