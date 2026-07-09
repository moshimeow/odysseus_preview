//! Planar-target pose bootstrap (homography → SE3).
//!
//! Given a [`BoardObservation`] (2D pixel ↔ 3D board-point pairs, all on the
//! z=0 plane of the board frame) and an *initial* camera model, recover an
//! approximate `board_t_camera` (camera pose in board frame) to seed
//! nonlinear calibration.
//!
//! Pipeline:
//! 1. Unproject pixels through the camera model and divide by ray-z to get
//!    normalized-image-plane coordinates `(xn, yn)`. This converts an
//!    arbitrary central-projection model into the equivalent pinhole problem.
//! 2. DLT homography fit on `(Xb, Yb) ↔ (xn, yn)`.
//! 3. Decompose H into `[r1 | r2 | t]`, recover `r3 = r1 × r2`, project the
//!    rotation onto SO(3) via SVD.
//! 4. Disambiguate sign so the board lies in front of the camera.
//!
//! Output is `board_t_camera`, the format `calibrate_double_sphere` expects.
//!
//! This is a *seed* — closed-form, no iteration. Residual error after this
//! step is taken care of by the joint nonlinear calibration.
//!
//! Pinhole approximation note: the unproject-then-divide-by-z step is exact
//! for any central projection model when the model is correct. With an
//! initial guess that's off by a few percent (typical), the recovered
//! homography is biased only at oblique corners; the residual is well
//! within the convergence basin of the downstream solver.

use crate::calibration::BoardObservation;
use crate::CameraModel;
use odysseus_solver::math3d::{SE3, SO3};
use nalgebra::{DMatrix, Matrix3};
use odysseus_solver::math3d::{Mat3, Vec3};

/// Failure modes for [`estimate_board_pose`]. All cases mean "use a different
/// frame or fix your detector" — there's no useful recovery the caller can do.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlanarPoseError {
    /// Fewer than 4 corners — homography needs at least 4 non-collinear points.
    TooFewCorners,
    /// SVD of the DLT system failed (degenerate / NaN inputs).
    DltSvdFailed,
    /// All unprojected rays had non-positive z (camera looking the wrong way?).
    NoValidRays,
    /// Recovered translation places the board behind the camera even after
    /// sign disambiguation — usually a sign of badly mismatched intrinsics.
    BoardBehindCamera,
}

/// Estimate `board_t_camera` (camera pose in board frame) from a single
/// planar-target observation and an initial camera model.
pub fn estimate_board_pose<C>(
    camera: &C,
    observation: &BoardObservation,
) -> Result<SE3<f64>, PlanarPoseError>
where
    C: CameraModel<f64>,
{
    if observation.corners.len() < 4 {
        return Err(PlanarPoseError::TooFewCorners);
    }

    // Step 1: pixel → normalized image plane (xn, yn) via the camera model.
    let mut pts: Vec<(f64, f64, f64, f64)> = Vec::with_capacity(observation.corners.len());
    for c in &observation.corners {
        let ray = camera.unproject(c.image_xy[0], c.image_xy[1]);
        if !(ray.z > 0.0) {
            continue;
        }
        let xn = ray.x / ray.z;
        let yn = ray.y / ray.z;
        // Board points are planar in z=0; we only need (Xb, Yb).
        pts.push((c.board_xyz.x, c.board_xyz.y, xn, yn));
    }
    if pts.len() < 4 {
        return Err(PlanarPoseError::NoValidRays);
    }

    // Step 2: DLT — solve A h = 0, where each correspondence contributes 2 rows.
    let n = pts.len();
    let mut a = DMatrix::<f64>::zeros(2 * n, 9);
    for (i, &(big_x, big_y, xn, yn)) in pts.iter().enumerate() {
        let r0 = 2 * i;
        let r1 = 2 * i + 1;
        a[(r0, 0)] = big_x;       a[(r0, 1)] = big_y;       a[(r0, 2)] = 1.0;
        a[(r0, 6)] = -xn * big_x; a[(r0, 7)] = -xn * big_y; a[(r0, 8)] = -xn;
        a[(r1, 3)] = big_x;       a[(r1, 4)] = big_y;       a[(r1, 5)] = 1.0;
        a[(r1, 6)] = -yn * big_x; a[(r1, 7)] = -yn * big_y; a[(r1, 8)] = -yn;
    }
    let svd = a.svd(false, true);
    let v_t = svd.v_t.ok_or(PlanarPoseError::DltSvdFailed)?;
    // Last row of V^T (= last column of V) = right singular vector of smallest
    // singular value = null-space solution.
    let h_row = v_t.row(v_t.nrows() - 1);
    let h = Matrix3::new(
        h_row[0], h_row[1], h_row[2],
        h_row[3], h_row[4], h_row[5],
        h_row[6], h_row[7], h_row[8],
    );

    // Step 3: decompose H = [r1 | r2 | t] (up to scale). Columns of H.
    let h1 = Vec3::new(h[(0, 0)], h[(1, 0)], h[(2, 0)]);
    let h2 = Vec3::new(h[(0, 1)], h[(1, 1)], h[(2, 1)]);
    let h3 = Vec3::new(h[(0, 2)], h[(1, 2)], h[(2, 2)]);
    let n1 = h1.norm();
    let n2 = h2.norm();
    if !(n1 > 0.0 && n2 > 0.0) {
        return Err(PlanarPoseError::DltSvdFailed);
    }
    // Average the two column norms to recover the scale; either alone is noisy.
    let scale = 2.0 / (n1 + n2);
    let mut r1 = h1 * scale;
    let mut r2 = h2 * scale;
    let mut t = h3 * scale;

    // Step 4: cheirality — board should be in front of camera (positive depth).
    if t.z < 0.0 {
        r1 = r1 * -1.0;
        r2 = r2 * -1.0;
        t = t * -1.0;
    }
    if t.z <= 0.0 {
        return Err(PlanarPoseError::BoardBehindCamera);
    }

    // Project [r1 | r2 | r1×r2] onto SO(3) via SVD: R = U * diag(1,1,det) * V^T.
    let r3 = r1.cross(r2);
    let r_raw = Matrix3::new(
        r1.x, r2.x, r3.x,
        r1.y, r2.y, r3.y,
        r1.z, r2.z, r3.z,
    );
    let r_svd = r_raw.svd(true, true);
    let u = r_svd.u.ok_or(PlanarPoseError::DltSvdFailed)?;
    let vt = r_svd.v_t.ok_or(PlanarPoseError::DltSvdFailed)?;
    let mut r = u * vt;
    if r.determinant() < 0.0 {
        // Flip sign of the last column of U so det(R) = +1.
        let mut u_fixed = u;
        for i in 0..3 {
            u_fixed[(i, 2)] = -u_fixed[(i, 2)];
        }
        r = u_fixed * vt;
    }
    let rot_mat: Mat3<f64> = r.into();
    let camera_t_board =
        SE3::from_rotation_translation(SO3::from_matrix(rot_mat), t);

    // We solved for camera_t_board (board points → camera frame); invert.
    Ok(camera_t_board.inverse())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::calibration::{AprilGridLayout, CornerObservation};
    use crate::DoubleSphereCamera;
    use approx::assert_abs_diff_eq;

    fn synthesize(
        camera: &DoubleSphereCamera<f64>,
        layout: &AprilGridLayout,
        board_t_camera: SE3<f64>,
    ) -> BoardObservation {
        let camera_t_board = board_t_camera.inverse();
        let mut obs = BoardObservation::new(0);
        for (tag_id, corner_idx, pos_board) in layout.iter_corners() {
            let pos_cam: Vec3<f64> = camera_t_board.transform_point(pos_board);
            if pos_cam.z <= 0.0 {
                continue;
            }
            let (u, v) = camera.project(pos_cam);
            obs.corners.push(CornerObservation {
                key: (tag_id, corner_idx),
                image_xy: [u, v],
                board_xyz: pos_board,
            });
        }
        obs
    }

    fn approx_se3_eq(a: &SE3<f64>, b: &SE3<f64>, eps: f64) {
        let delta = a.inverse() * *b;
        let tan = delta.log();
        assert_abs_diff_eq!(tan.rotation.x, 0.0, epsilon = eps);
        assert_abs_diff_eq!(tan.rotation.y, 0.0, epsilon = eps);
        assert_abs_diff_eq!(tan.rotation.z, 0.0, epsilon = eps);
        assert_abs_diff_eq!(tan.translation.x, 0.0, epsilon = eps);
        assert_abs_diff_eq!(tan.translation.y, 0.0, epsilon = eps);
        assert_abs_diff_eq!(tan.translation.z, 0.0, epsilon = eps);
    }

    #[test]
    fn recovers_frontal_pose() {
        let camera = DoubleSphereCamera::pinhole_equivalent(600.0, 600.0, 320.0, 240.0);
        let layout = AprilGridLayout::new(6, 6, 0.0392, 0.3);
        // Camera 0.5 m in front of board centre, looking back at it.
        let board_center = Vec3::new(0.15, 0.15, 0.0);
        let cam_pos_board = board_center + Vec3::new(0.0, 0.0, 0.5);
        // board_t_camera: rotation flips z so camera's +z points at the board.
        let rot = SO3::from_matrix(Mat3::from_cols(
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, -1.0, 0.0),
            Vec3::new(0.0, 0.0, -1.0),
        ));
        let board_t_camera = SE3::from_rotation_translation(rot, cam_pos_board);

        let obs = synthesize(&camera, &layout, board_t_camera);
        assert!(obs.corners.len() >= 4);
        let est = estimate_board_pose(&camera, &obs).unwrap();
        approx_se3_eq(&est, &board_t_camera, 1e-6);
    }

    #[test]
    fn recovers_tilted_pose() {
        let camera = DoubleSphereCamera::pinhole_equivalent(600.0, 600.0, 320.0, 240.0);
        let layout = AprilGridLayout::new(6, 6, 0.0392, 0.3);
        // Frontal pose composed with a small rotation around board's x axis.
        let cam_pos_board = Vec3::new(0.15, 0.15, 0.5);
        let base_rot = Mat3::from_cols(
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, -1.0, 0.0),
            Vec3::new(0.0, 0.0, -1.0),
        );
        let theta = 0.35_f64;
        let tilt = Mat3::from_cols(
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, theta.cos(), theta.sin()),
            Vec3::new(0.0, -theta.sin(), theta.cos()),
        );
        let rot_mat = Mat3 {
            x_axis: Vec3::new(
                base_rot.x_axis.x * tilt.x_axis.x + base_rot.y_axis.x * tilt.x_axis.y + base_rot.z_axis.x * tilt.x_axis.z,
                base_rot.x_axis.y * tilt.x_axis.x + base_rot.y_axis.y * tilt.x_axis.y + base_rot.z_axis.y * tilt.x_axis.z,
                base_rot.x_axis.z * tilt.x_axis.x + base_rot.y_axis.z * tilt.x_axis.y + base_rot.z_axis.z * tilt.x_axis.z,
            ),
            y_axis: Vec3::new(
                base_rot.x_axis.x * tilt.y_axis.x + base_rot.y_axis.x * tilt.y_axis.y + base_rot.z_axis.x * tilt.y_axis.z,
                base_rot.x_axis.y * tilt.y_axis.x + base_rot.y_axis.y * tilt.y_axis.y + base_rot.z_axis.y * tilt.y_axis.z,
                base_rot.x_axis.z * tilt.y_axis.x + base_rot.y_axis.z * tilt.y_axis.y + base_rot.z_axis.z * tilt.y_axis.z,
            ),
            z_axis: Vec3::new(
                base_rot.x_axis.x * tilt.z_axis.x + base_rot.y_axis.x * tilt.z_axis.y + base_rot.z_axis.x * tilt.z_axis.z,
                base_rot.x_axis.y * tilt.z_axis.x + base_rot.y_axis.y * tilt.z_axis.y + base_rot.z_axis.y * tilt.z_axis.z,
                base_rot.x_axis.z * tilt.z_axis.x + base_rot.y_axis.z * tilt.z_axis.y + base_rot.z_axis.z * tilt.z_axis.z,
            ),
        };
        let board_t_camera =
            SE3::from_rotation_translation(SO3::from_matrix(rot_mat), cam_pos_board);

        let obs = synthesize(&camera, &layout, board_t_camera);
        assert!(obs.corners.len() >= 4);
        let est = estimate_board_pose(&camera, &obs).unwrap();
        approx_se3_eq(&est, &board_t_camera, 1e-6);
    }

    #[test]
    fn recovers_with_fisheye_distortion() {
        // Real DS camera (xi, alpha non-zero). The unproject-then-divide-by-z
        // step is exact, so we should still recover essentially exactly.
        let camera = DoubleSphereCamera::new(280.0, 280.0, 320.0, 240.0, -0.2, 0.58);
        let layout = AprilGridLayout::new(6, 6, 0.0392, 0.3);
        let cam_pos_board = Vec3::new(0.15, 0.15, 0.4);
        let rot = SO3::from_matrix(Mat3::from_cols(
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, -1.0, 0.0),
            Vec3::new(0.0, 0.0, -1.0),
        ));
        let board_t_camera = SE3::from_rotation_translation(rot, cam_pos_board);
        let obs = synthesize(&camera, &layout, board_t_camera);
        assert!(obs.corners.len() >= 4);
        let est = estimate_board_pose(&camera, &obs).unwrap();
        approx_se3_eq(&est, &board_t_camera, 1e-5);
    }

    #[test]
    fn fails_with_too_few_corners() {
        let camera = DoubleSphereCamera::pinhole_equivalent(600.0, 600.0, 320.0, 240.0);
        let mut obs = BoardObservation::new(0);
        obs.corners.push(CornerObservation {
            key: (0, 0),
            image_xy: [100.0, 100.0],
            board_xyz: Vec3::new(0.0, 0.0, 0.0),
        });
        assert_eq!(
            estimate_board_pose(&camera, &obs),
            Err(PlanarPoseError::TooFewCorners)
        );
    }
}
