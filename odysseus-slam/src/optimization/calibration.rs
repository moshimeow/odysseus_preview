//! Camera intrinsic calibration via AprilGrid observations.
//!
//! Currently specialized to the [`DoubleSphereCamera`] model. The structure
//! generalizes — `compose_params!` defines the per-model intrinsics struct and
//! the optimizer is otherwise model-agnostic — but only DS is wired up here.
//! Add other models by cloning `DoubleSphereIntrinsics` + `calibrate_double_sphere`.
//!
//! The optimizer minimizes Σ ||project(K, T_cam_board · p_board) − pixel||²
//! over (K, {T_board_camera}) using Levenberg–Marquardt with a Huber loss on
//! each scalar residual.

use crate::calibration::BoardObservation;
use crate::camera::DoubleSphereCamera;
use crate::math::{SE3, SO3};
use nalgebra::DVector;
use odysseus_solver::compose_params;
use odysseus_solver::math3d::Vec3;
use odysseus_solver::sparse_solver::SparseLevenbergMarquardt;
use odysseus_solver::{Jet, Real};
use std::collections::HashMap;

use super::apply_huber_loss;

// ---------------------------------------------------------------------------
// Intrinsics struct via compose_params!
// ---------------------------------------------------------------------------

compose_params! {
    /// Double Sphere intrinsics laid out for `compose_params!`-driven packing.
    /// `From<[T; 6]>` packs in the order `[fx, fy, cx, cy, xi, alpha]`.
    #[derive(Debug, Clone, Copy)]
    pub struct DoubleSphereIntrinsics<T> {
        focal_principal: [T; 4] [4],   // fx, fy, cx, cy
        distortion: [T; 2] [2],        // xi, alpha
    }
}

impl<T: Copy> DoubleSphereIntrinsics<T> {
    pub fn fx(&self) -> T { self.focal_principal[0] }
    pub fn fy(&self) -> T { self.focal_principal[1] }
    pub fn cx(&self) -> T { self.focal_principal[2] }
    pub fn cy(&self) -> T { self.focal_principal[3] }
    pub fn xi(&self) -> T { self.distortion[0] }
    pub fn alpha(&self) -> T { self.distortion[1] }
}

impl<T: Real> DoubleSphereIntrinsics<T>
where
    T::Scalar: PartialOrd<f64>,
{
    pub fn to_camera(&self) -> DoubleSphereCamera<T> {
        DoubleSphereCamera::new(
            self.fx(), self.fy(), self.cx(), self.cy(), self.xi(), self.alpha(),
        )
    }
}

impl<T: Copy> From<&DoubleSphereCamera<T>> for DoubleSphereIntrinsics<T> {
    fn from(c: &DoubleSphereCamera<T>) -> Self {
        Self {
            focal_principal: [c.fx, c.fy, c.cx, c.cy],
            distortion: [c.xi, c.alpha],
        }
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct DoubleSphereCalibConfig {
    pub max_iterations: usize,
    pub tolerance: f64,
    /// Pixels. Residuals beyond this are downweighted (Huber).
    pub huber_delta_pixels: f64,
    pub verbose: bool,
}

impl Default for DoubleSphereCalibConfig {
    fn default() -> Self {
        Self {
            max_iterations: 50,
            tolerance: 1e-12,
            huber_delta_pixels: 1.5,
            verbose: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct DoubleSphereCalibResult {
    pub camera: DoubleSphereCamera<f64>,
    /// Recovered camera-pose-in-board-frame for each input `frame_idx`.
    pub board_t_camera: HashMap<usize, SE3<f64>>,
    pub final_rms_pixel_error: f64,
    pub iterations: usize,
}

// ---------------------------------------------------------------------------
// Solver
// ---------------------------------------------------------------------------

const N_INT: usize = 6;
const N_EXT: usize = 6;
const N_DERIV: usize = N_INT + N_EXT;
const N_RES_PER_OBS: usize = 2;

/// Calibrate a Double Sphere camera from per-frame AprilGrid observations.
///
/// `initial_extrinsics` is the camera-pose-in-board-frame guess for each
/// `frame_idx` appearing in `observations`. Frames missing from the map default
/// to `SE3::identity()`, which is rarely a good guess — supply a real init
/// (e.g. from P3P / homography) for non-trivial cases.
pub fn calibrate_double_sphere(
    initial_camera: DoubleSphereCamera<f64>,
    initial_extrinsics: &HashMap<usize, SE3<f64>>,
    observations: &[BoardObservation],
    config: &DoubleSphereCalibConfig,
) -> DoubleSphereCalibResult {
    // Frame-id mapping. Local indices are 0..n_frames in sorted order.
    let mut frames: Vec<usize> = observations.iter().map(|o| o.frame_idx).collect();
    frames.sort_unstable();
    frames.dedup();
    let frame_to_local: HashMap<usize, usize> =
        frames.iter().enumerate().map(|(i, &f)| (f, i)).collect();
    let n_frames = frames.len();

    // Flatten observations.
    let mut obs_frame_local: Vec<usize> = Vec::new();
    let mut obs_pixel: Vec<[f64; 2]> = Vec::new();
    let mut obs_board_xyz: Vec<Vec3<f64>> = Vec::new();
    for board_obs in observations {
        let f_local = frame_to_local[&board_obs.frame_idx];
        for c in &board_obs.corners {
            obs_frame_local.push(f_local);
            obs_pixel.push(c.image_xy);
            obs_board_xyz.push(c.board_xyz);
        }
    }
    let n_obs = obs_frame_local.len();
    let n_residuals = n_obs * N_RES_PER_OBS;
    let n_params = N_INT + n_frames * N_EXT;

    // Pack initial parameters.
    let mut params = DVector::<f64>::zeros(n_params);
    let init_int_arr: [f64; N_INT] = DoubleSphereIntrinsics::from(&initial_camera).into();
    params.as_mut_slice()[..N_INT].copy_from_slice(&init_int_arr);
    for (&frame, &local) in &frame_to_local {
        let pose = initial_extrinsics
            .get(&frame)
            .copied()
            .unwrap_or_else(SE3::identity);
        let omega = pose.rotation.log();
        let t = pose.translation;
        let base = N_INT + local * N_EXT;
        params[base] = omega.x;
        params[base + 1] = omega.y;
        params[base + 2] = omega.z;
        params[base + 3] = t.x;
        params[base + 4] = t.y;
        params[base + 5] = t.z;
    }

    // Sparsity: 2 rows per observation, each touching all intrinsic columns plus
    // the corresponding frame's 6 extrinsic columns. Row-major.
    let mut entries: Vec<(usize, usize)> = Vec::with_capacity(n_residuals * N_DERIV);
    for (obs_i, &f_local) in obs_frame_local.iter().enumerate() {
        let row_base = obs_i * N_RES_PER_OBS;
        let ext_base = N_INT + f_local * N_EXT;
        for r in 0..N_RES_PER_OBS {
            let row = row_base + r;
            for c in 0..N_INT {
                entries.push((row, c));
            }
            for c in 0..N_EXT {
                entries.push((row, ext_base + c));
            }
        }
    }

    let mut solver = SparseLevenbergMarquardt::<f64>::new(n_residuals, n_params, &entries)
        .with_tolerance(config.tolerance)
        .with_max_iterations(config.max_iterations)
        .with_verbose(config.verbose);

    let mut iter_count = 0usize;
    let huber = config.huber_delta_pixels;

    let final_params = solver.solve(
        params,
        |params, residuals, jac_data| {
            // Build a single Jet'd intrinsics struct shared across all observations
            // (it doesn't depend on which observation we're processing).
            let intrinsics_jet = DoubleSphereIntrinsics::<Jet<f64, N_DERIV>> {
                focal_principal: [
                    Jet::variable(params[0], 0),
                    Jet::variable(params[1], 1),
                    Jet::variable(params[2], 2),
                    Jet::variable(params[3], 3),
                ],
                distortion: [
                    Jet::variable(params[4], 4),
                    Jet::variable(params[5], 5),
                ],
            };
            let camera_jet = intrinsics_jet.to_camera();

            for obs_i in 0..n_obs {
                let f_local = obs_frame_local[obs_i];
                let ext_base = N_INT + f_local * N_EXT;

                // Per-frame extrinsic: rotation as axis-angle, translation direct.
                // Deriv indices 6..12.
                let omega = Vec3::new(
                    Jet::<f64, N_DERIV>::variable(params[ext_base], 6),
                    Jet::<f64, N_DERIV>::variable(params[ext_base + 1], 7),
                    Jet::<f64, N_DERIV>::variable(params[ext_base + 2], 8),
                );
                let t = Vec3::new(
                    Jet::<f64, N_DERIV>::variable(params[ext_base + 3], 9),
                    Jet::<f64, N_DERIV>::variable(params[ext_base + 4], 10),
                    Jet::<f64, N_DERIV>::variable(params[ext_base + 5], 11),
                );
                let board_t_camera = SE3::from_rotation_translation(SO3::exp(omega), t);
                let camera_t_board = board_t_camera.inverse();

                // Board point as constant.
                let xyz = obs_board_xyz[obs_i];
                let pt_board = Vec3::new(
                    Jet::<f64, N_DERIV>::constant(xyz.x),
                    Jet::<f64, N_DERIV>::constant(xyz.y),
                    Jet::<f64, N_DERIV>::constant(xyz.z),
                );
                let pt_cam = camera_t_board.transform_point(pt_board);
                let (u_pred, v_pred) = camera_jet.project(pt_cam);

                let [u_obs, v_obs] = obs_pixel[obs_i];
                let row_base = obs_i * N_RES_PER_OBS;
                let jac_base = row_base * N_DERIV;

                // Row 0: u-residual.
                let mut r_u = u_pred.value - u_obs;
                let mut jac_u: [f64; N_DERIV] = u_pred.derivs;
                apply_huber_loss(huber, &mut r_u, &mut jac_u);
                residuals[row_base] = r_u;
                jac_data[jac_base..jac_base + N_DERIV].copy_from_slice(&jac_u);

                // Row 1: v-residual.
                let mut r_v = v_pred.value - v_obs;
                let mut jac_v: [f64; N_DERIV] = v_pred.derivs;
                apply_huber_loss(huber, &mut r_v, &mut jac_v);
                residuals[row_base + 1] = r_v;
                jac_data[jac_base + N_DERIV..jac_base + 2 * N_DERIV].copy_from_slice(&jac_v);
            }
        },
        |it, _result, _params| {
            iter_count = it + 1;
        },
    );

    // Unpack.
    let mut int_arr = [0.0_f64; N_INT];
    int_arr.copy_from_slice(&final_params.as_slice()[..N_INT]);
    let camera_final: DoubleSphereCamera<f64> =
        DoubleSphereIntrinsics::<f64>::from(int_arr).to_camera();

    let mut board_t_camera: HashMap<usize, SE3<f64>> = HashMap::with_capacity(n_frames);
    for (&frame, &local) in &frame_to_local {
        let base = N_INT + local * N_EXT;
        let omega = Vec3::new(
            final_params[base],
            final_params[base + 1],
            final_params[base + 2],
        );
        let t = Vec3::new(
            final_params[base + 3],
            final_params[base + 4],
            final_params[base + 5],
        );
        board_t_camera.insert(frame, SE3::from_rotation_translation(SO3::exp(omega), t));
    }

    // RMS pixel error using the recovered camera (un-Huber'd).
    let mut sum_sq = 0.0_f64;
    for obs_i in 0..n_obs {
        let f_local = obs_frame_local[obs_i];
        let frame = frames[f_local];
        let pose = board_t_camera[&frame];
        let camera_t_board = pose.inverse();
        let pt_cam = camera_t_board.transform_point(obs_board_xyz[obs_i]);
        let (u, v) = camera_final.project(pt_cam);
        let [u_obs, v_obs] = obs_pixel[obs_i];
        sum_sq += (u - u_obs).powi(2) + (v - v_obs).powi(2);
    }
    let final_rms_pixel_error = (sum_sq / (2.0 * n_obs as f64)).sqrt();

    DoubleSphereCalibResult {
        camera: camera_final,
        board_t_camera,
        final_rms_pixel_error,
        iterations: iter_count,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::calibration::{AprilGridLayout, BoardObservation, CornerObservation};
    use approx::assert_abs_diff_eq;
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    /// Project the layout's corners through `camera` from each pose, returning
    /// only those that land inside `[0, image_w) × [0, image_h)` and in front
    /// of the camera. Pose convention: `board_t_camera` (camera in board frame).
    fn synth_observations(
        camera: &DoubleSphereCamera<f64>,
        layout: &AprilGridLayout,
        board_t_camera_per_frame: &[SE3<f64>],
        image_w: f64,
        image_h: f64,
    ) -> Vec<BoardObservation> {
        let mut out = Vec::new();
        for (frame_idx, board_t_camera) in board_t_camera_per_frame.iter().enumerate() {
            let camera_t_board = board_t_camera.inverse();
            let mut bo = BoardObservation::new(frame_idx);
            for (tag_id, corner_idx, board_xyz) in layout.iter_corners() {
                let pt_cam = camera_t_board.transform_point(board_xyz);
                if pt_cam.z <= 0.0 {
                    continue;
                }
                let (u, v) = camera.project(pt_cam);
                if !u.is_finite() || !v.is_finite() {
                    continue;
                }
                if u < 0.0 || u >= image_w || v < 0.0 || v >= image_h {
                    continue;
                }
                bo.corners.push(CornerObservation {
                    key: (tag_id, corner_idx),
                    image_xy: [u, v],
                    board_xyz,
                });
            }
            if !bo.corners.is_empty() {
                out.push(bo);
            }
        }
        out
    }

    #[test]
    fn test_synthetic_recovery_double_sphere() {
        // Ground truth.
        let truth = DoubleSphereCamera::new(500.0, 500.0, 320.0, 240.0, -0.2, 0.5);
        let layout = AprilGridLayout::new(6, 6, 0.05, 0.3);

        // 20 random camera poses, all looking at the board.
        // Convention: pose is `board_t_camera` (camera-in-board-frame).
        // Camera axes (CV): +X right, +Y down, +Z forward. Board axes:
        // +X right, +Y up, +Z out of board. With camera rotation = identity
        // and camera at (W/2, H/2, -d), the board is at +d in camera Z and
        // visible (image just appears upside-down — irrelevant for calibration).
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let mut poses = Vec::new();
        let board_w = layout.tag_pitch_m() * (layout.tag_cols as f64 - 1.0) + layout.tag_size_m;
        let board_h = layout.tag_pitch_m() * (layout.tag_rows as f64 - 1.0) + layout.tag_size_m;
        let centre = Vec3::new(board_w * 0.5, board_h * 0.5, 0.0);

        for _ in 0..20 {
            let depth: f64 = rng.gen_range(0.5..1.2);
            let dx: f64 = rng.gen_range(-0.05..0.05);
            let dy: f64 = rng.gen_range(-0.05..0.05);
            let cam_pos = Vec3::new(centre.x + dx, centre.y + dy, -depth);
            let perturb_axis = Vec3::new(
                rng.gen_range(-0.3..0.3),
                rng.gen_range(-0.3..0.3),
                rng.gen_range(-0.2..0.2),
            );
            let rotation = SO3::exp(perturb_axis);
            poses.push(SE3::from_rotation_translation(rotation, cam_pos));
        }

        let observations = synth_observations(&truth, &layout, &poses, 640.0, 480.0);
        let total_corners: usize = observations.iter().map(|o| o.corners.len()).sum();
        assert!(
            total_corners > 200,
            "synthetic setup should produce plenty of corners; got {total_corners}"
        );

        // Initial guess: intrinsics off by a few percent, extrinsics at truth
        // (the test isolates intrinsic-recovery quality; a pose-init test is
        // separate scope).
        let init_camera = DoubleSphereCamera::new(525.0, 480.0, 305.0, 255.0, -0.10, 0.40);
        let init_ext: HashMap<usize, SE3<f64>> =
            poses.iter().enumerate().map(|(i, p)| (i, *p)).collect();

        let mut config = DoubleSphereCalibConfig::default();
        config.verbose = false;

        let result = calibrate_double_sphere(init_camera, &init_ext, &observations, &config);

        assert_abs_diff_eq!(result.camera.fx, truth.fx, epsilon = 1e-3);
        assert_abs_diff_eq!(result.camera.fy, truth.fy, epsilon = 1e-3);
        assert_abs_diff_eq!(result.camera.cx, truth.cx, epsilon = 1e-3);
        assert_abs_diff_eq!(result.camera.cy, truth.cy, epsilon = 1e-3);
        assert_abs_diff_eq!(result.camera.xi, truth.xi, epsilon = 1e-4);
        assert_abs_diff_eq!(result.camera.alpha, truth.alpha, epsilon = 1e-4);
        assert!(
            result.final_rms_pixel_error < 1e-6,
            "RMS pixel error too high: {}",
            result.final_rms_pixel_error
        );
    }

    #[test]
    fn test_compose_params_roundtrip() {
        let camera = DoubleSphereCamera::new(500.0, 600.0, 320.0, 240.0, -0.2, 0.5);
        let intr: DoubleSphereIntrinsics<f64> = (&camera).into();
        let arr: [f64; 6] = intr.into();
        assert_eq!(arr, [500.0, 600.0, 320.0, 240.0, -0.2, 0.5]);

        let intr_back = DoubleSphereIntrinsics::<f64>::from(arr);
        let cam_back = intr_back.to_camera();
        assert_abs_diff_eq!(cam_back.fx, 500.0, epsilon = 1e-12);
        assert_abs_diff_eq!(cam_back.alpha, 0.5, epsilon = 1e-12);
    }
}
