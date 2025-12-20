//! 3D Pointcloud Alignment with Rerun Visualization
//!
//! Full 6-DOF pose estimation with interactive 3D visualization

use nalgebra::SVector;
use odysseus_solver::{
    math3d::{rodrigues_to_matrix, transform_point, Vec3},
    Jet, LevenbergMarquardt, Real,
};

#[cfg(feature = "visualization")]
use rerun as rr;

const N_PARAMS: usize = 6; // [tx, ty, tz, rx, ry, rz]
const N_POINTS: usize = 50;
const N_RESIDUALS: usize = N_POINTS * 3;

type Jet6 = Jet<f32, N_PARAMS>;
type Params = SVector<f32, N_PARAMS>;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎯 odysseus-solver: 3D Pointcloud Alignment");
    println!("============================================\n");

    #[cfg(feature = "visualization")]
    let rec = rr::RecordingStreamBuilder::new("pointcloud_alignment").spawn()?;

    // Generate pointcloud
    let source_points = generate_pointcloud(N_POINTS, 42);
    println!("Generated {} source points\n", N_POINTS);

    // True pose
    let true_pose = Params::new(0.3, 0.2, 0.1, -2.0, 1.5, 0.1);
    println!("🎯 True Pose:");
    println!("  Translation: [{:.3}, {:.3}, {:.3}]", true_pose[0], true_pose[1], true_pose[2]);
    println!("  Rotation:    [{:.3}, {:.3}, {:.3}]", true_pose[3], true_pose[4], true_pose[5]);

    // Generate target
    let target_points = transform_pointcloud(&source_points, &true_pose);

    // Initial guess
    let mut params = Params::zeros();
    println!("\n🚀 Initial Guess: all zeros\n");

    // Cost function
    let cost_fn = |params: &Params| {
        compute_residuals_and_jacobian(&source_points, &target_points, params)
    };

    // Solve with visualization
    let mut solver = LevenbergMarquardt::<f32, N_PARAMS, N_RESIDUALS>::new()
        .with_tolerance(1e-6)
        .with_max_iterations(30)
        .with_verbose(false);

    println!("🔧 Starting Optimization:");

    // Wrap the cost function to make it compatible with solve()
    let cost_fn_wrapped = |params: &Params,
                           residuals: &mut SVector<f32, N_RESIDUALS>,
                           jacobian: &mut nalgebra::SMatrix<f32, N_RESIDUALS, N_PARAMS>| {
        let (r, j) = cost_fn(params);
        *residuals = r;
        *jacobian = j;
    };

    params = solver.solve(params, cost_fn_wrapped, |iter, result, current_estimate| {
        if iter < 3 || iter % 5 == 0 {
            println!(
                "  Iter {:2}: error={:.6}, λ={:.2e}",
                iter, result.error, result.lambda
            );
        }

        #[cfg(feature = "visualization")]
        {
            rec.set_time_sequence("optimization_step", iter as i64);
            let current_transformed = transform_pointcloud(&source_points, current_estimate);
            log_pointclouds(&rec, &source_points, &target_points, &current_transformed);
        }
    });

    // Results
    println!("\n🎉 Final Results:");
    println!("  Estimated: [{:.6}, {:.6}, {:.6}, {:.6}, {:.6}, {:.6}]",
        params[0], params[1], params[2], params[3], params[4], params[5]);
    println!("  True:      [{:.6}, {:.6}, {:.6}, {:.6}, {:.6}, {:.6}]",
        true_pose[0], true_pose[1], true_pose[2], true_pose[3], true_pose[4], true_pose[5]);

    let translation_error = ((params[0] - true_pose[0]).powi(2)
        + (params[1] - true_pose[1]).powi(2)
        + (params[2] - true_pose[2]).powi(2))
    .sqrt();
    let rotation_error = ((params[3] - true_pose[3]).powi(2)
        + (params[4] - true_pose[4]).powi(2)
        + (params[5] - true_pose[5]).powi(2))
    .sqrt();

    println!("\n📊 Errors:");
    println!("  Translation: {:.2e}", translation_error);
    println!("  Rotation:    {:.2e}", rotation_error);

    #[cfg(feature = "visualization")]
    {
        // Log final
        rec.set_time_sequence("optimization_step", 30);
        let final_transformed = transform_pointcloud(&source_points, &params);
        log_pointclouds(&rec, &source_points, &target_points, &final_transformed);
        println!("\n🎬 Open the Rerun viewer to see the optimization!");
        println!("   - Source points: Blue");
        println!("   - Target points: Gray");
        println!("   - Current estimate: Orange");
        println!("   - Use the timeline slider to see each iteration!");
    }

    Ok(())
}

fn generate_pointcloud(n: usize, seed: u64) -> Vec<Vec3<f32>> {
    let mut rng = seed;
    let lcg = |r: &mut u64| -> f32 {
        *r = r.wrapping_mul(1103515245).wrapping_add(12345);
        ((*r / 65536) % 32768) as f32 / 32768.0 * 2.0 - 1.0
    };

    (0..n)
        .map(|_| Vec3::new(lcg(&mut rng), lcg(&mut rng), lcg(&mut rng)))
        .collect()
}

fn transform_pointcloud(points: &[Vec3<f32>], pose: &Params) -> Vec<Vec3<f32>> {
    transform_pointcloud_generic(points, pose)
}

fn transform_pointcloud_generic<T: Real<Scalar = f32>>(points: &[Vec3<f32>], pose: &SVector<T, N_PARAMS>) -> Vec<Vec3<T>> {
    let translation = Vec3::new(pose[0], pose[1], pose[2]);
    let rvec = Vec3::new(pose[3], pose[4], pose[5]);
    let rotation = rodrigues_to_matrix(rvec);

    points
        .iter()
        .map(|&p| {
            let pt = Vec3::new(
                T::constant(p.x),
                T::constant(p.y),
                T::constant(p.z),
            );
            transform_point(rotation, translation, pt)
        })
        .collect()
}

fn compute_residuals_and_jacobian(
    source: &[Vec3<f32>],
    target: &[Vec3<f32>],
    params: &Params,
) -> (SVector<f32, N_RESIDUALS>, nalgebra::SMatrix<f32, N_RESIDUALS, N_PARAMS>) {
    let pose_jet: SVector<Jet6, N_PARAMS> = SVector::from_fn(|i, _| Jet6::variable(params[i], i));
    let transformed = transform_pointcloud_generic(source, &pose_jet);

    let mut residuals = SVector::<f32, N_RESIDUALS>::zeros();
    let mut jacobian = nalgebra::SMatrix::<f32, N_RESIDUALS, N_PARAMS>::zeros();

    for (i, (transformed_pt, target_pt)) in transformed.iter().zip(target.iter()).enumerate() {
        let rx = transformed_pt.x - Jet6::constant(target_pt.x);
        let ry = transformed_pt.y - Jet6::constant(target_pt.y);
        let rz = transformed_pt.z - Jet6::constant(target_pt.z);

        residuals[i * 3 + 0] = rx.value;
        residuals[i * 3 + 1] = ry.value;
        residuals[i * 3 + 2] = rz.value;

        for j in 0..N_PARAMS {
            jacobian[(i * 3 + 0, j)] = rx.derivs[j];
            jacobian[(i * 3 + 1, j)] = ry.derivs[j];
            jacobian[(i * 3 + 2, j)] = rz.derivs[j];
        }
    }

    (residuals, jacobian)
}

#[cfg(feature = "visualization")]
fn log_pointcloud(
    rec: &rr::RecordingStream,
    entity_path: &str,
    points: &[Vec3<f32>],
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

#[cfg(feature = "visualization")]
fn log_pointclouds(
    rec: &rr::RecordingStream,
    source: &[Vec3<f32>],
    target: &[Vec3<f32>],
    current: &[Vec3<f32>],
) {
    log_pointcloud(rec, "world/source", source, [100, 100, 255], 0.03);
    log_pointcloud(rec, "world/target", target, [150, 150, 150], 0.02);
    log_pointcloud(rec, "world/current", current, [255, 150, 50], 0.025);
}
