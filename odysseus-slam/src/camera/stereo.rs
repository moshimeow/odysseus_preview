//! Stereo camera pair (generic over the underlying intrinsic model).
//!
//! `StereoCamera<C, T>` holds two cameras of model type `C` separated by a
//! horizontal baseline along +X.  `C` can be any `CameraModel<T>` implementor:
//! `PinholeCamera<T>`, `DoubleSphereCamera<T>`, etc.
//!
//! Standard stereo convention:
//! - Left camera at the origin
//! - Right camera at `(baseline, 0, 0)` (translated along +X)
//! - Both cameras looking along +Z

use crate::camera::{CameraConstantJet, CameraModel, PinholeCamera};
use odysseus_solver::math3d::Vec3;
use odysseus_solver::{Jet, Real};

#[derive(Debug, Clone, Copy)]
pub struct StereoCamera<C, T> {
    pub left: C,
    pub right: C,
    /// Baseline distance (metres). Right camera is at `(baseline, 0, 0)` in
    /// the left camera's frame.
    pub baseline: T,
}

impl<T: Real, C: CameraModel<T>> StereoCamera<C, T> {
    /// Build a stereo pair from a single camera (cloned for both eyes) and a
    /// baseline.
    pub fn new(camera: C, baseline: T) -> Self {
        Self {
            left: camera,
            right: camera,
            baseline,
        }
    }

    /// Build from explicit left/right intrinsics (rare — use `new` if both
    /// eyes share intrinsics, which is the standard rectified setup).
    pub fn new_split(left: C, right: C, baseline: T) -> Self {
        Self { left, right, baseline }
    }

    /// Project a 3D point (left-camera frame) to both image planes.
    /// Returns `(u_left, v_left, u_right, v_right)`.
    pub fn project_stereo(&self, point_left: Vec3<T>) -> (T, T, T, T) {
        let (u_left, v_left) = self.left.project(point_left);
        let point_right = Vec3::new(
            point_left.x - self.baseline,
            point_left.y,
            point_left.z,
        );
        let (u_right, v_right) = self.right.project(point_right);
        (u_left, v_left, u_right, v_right)
    }
}

/// Convenience constructor for the common pinhole + square-image case.
/// Pinhole-only because `simple` requires a notion of focal length, which
/// other models parameterize differently.
impl<T: Real> StereoCamera<PinholeCamera<T>, T> {
    pub fn simple(focal_length: T, image_width: T, image_height: T, baseline: T) -> Self {
        let camera = PinholeCamera::simple(focal_length, image_width, image_height);
        Self::new(camera, baseline)
    }
}

/// Lift an `f64`-valued stereo pair into a `Jet`-valued one with constant
/// intrinsics.  Used when the tracker autodiffs a residual w.r.t. the 3D
/// point but treats camera intrinsics + baseline as fixed.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera::PinholeCamera;
    use approx::assert_abs_diff_eq;

    #[test]
    fn triangulate_midpoint_matches_disparity_formula() {
        // Rectified pinhole: midpoint triangulation must equal the closed-form
        // disparity solution.
        let cam = PinholeCamera::<f64>::new(500.0, 500.0, 320.0, 240.0);
        let stereo = StereoCamera::new(cam, 0.064);

        let world_pt = Vec3::new(0.10, -0.05, 1.5);
        let (u_l, v_l) = cam.project(world_pt);
        let world_pt_right = Vec3::new(world_pt.x - 0.064, world_pt.y, world_pt.z);
        let (u_r, v_r) = cam.project(world_pt_right);

        let triangulated = stereo.triangulate_midpoint(u_l, v_l, u_r, v_r).unwrap();
        assert_abs_diff_eq!(triangulated.x, world_pt.x, epsilon = 1e-9);
        assert_abs_diff_eq!(triangulated.y, world_pt.y, epsilon = 1e-9);
        assert_abs_diff_eq!(triangulated.z, world_pt.z, epsilon = 1e-9);
    }
}

impl<C: CameraConstantJet> StereoCamera<C, f64> {
    pub fn constant_jet<const N: usize>(&self) -> StereoCamera<C::Jet<N>, Jet<f64, N>> {
        StereoCamera {
            left: self.left.constant_jet::<N>(),
            right: self.right.constant_jet::<N>(),
            baseline: Jet::constant(self.baseline),
        }
    }
}

impl<C: CameraModel<f64>> StereoCamera<C, f64> {
    /// Triangulate a stereo correspondence to a 3D point in the **left camera
    /// frame**.  Uses midpoint triangulation: unproject each pixel to a ray,
    /// translate the right ray's origin by `(baseline, 0, 0)`, and find the
    /// 3D point closest to both rays in a least-squares sense.
    ///
    /// This is the camera-model-agnostic generalization of the rectified
    /// pinhole shortcut `Z = baseline * fx / disparity`.  For a rectified
    /// pinhole it produces the same result; for distorted models (DS, KB4)
    /// it does the geometrically correct thing.
    ///
    /// Returns `None` if either ray points away from the scene, the rays are
    /// near-parallel, or the closest-approach point lies behind a camera.
    pub fn triangulate_midpoint(
        &self,
        u_left: f64,
        v_left: f64,
        u_right: f64,
        v_right: f64,
    ) -> Option<Vec3<f64>> {
        let d_left = self.left.unproject(u_left, v_left);
        let d_right_local = self.right.unproject(u_right, v_right);
        // Right ray expressed in the left frame: shared orientation, origin
        // shifted by +baseline along X.
        let origin_right = Vec3::new(self.baseline, 0.0, 0.0);

        // Solve for (t_l, t_r) minimizing ||t_l * d_l - origin_right - t_r * d_r||².
        // Normal equations:
        //   [ d_l·d_l   -d_l·d_r ] [t_l]   [ d_l·b ]
        //   [-d_l·d_r    d_r·d_r ] [t_r] = [-d_r·b ]
        // where b = origin_right.
        let dot = |a: Vec3<f64>, b: Vec3<f64>| a.x * b.x + a.y * b.y + a.z * b.z;
        let a11 = dot(d_left, d_left);
        let a22 = dot(d_right_local, d_right_local);
        let a12 = -dot(d_left, d_right_local);
        let det = a11 * a22 - a12 * a12;
        if det.abs() < 1e-12 {
            // Near-parallel rays — large depth uncertainty.
            return None;
        }
        let b1 = dot(d_left, origin_right);
        let b2 = -dot(d_right_local, origin_right);
        let t_l = (a22 * b1 - a12 * b2) / det;
        let t_r = (a11 * b2 - a12 * b1) / det;
        if t_l <= 0.0 || t_r <= 0.0 {
            return None;
        }
        let p_left = Vec3::new(d_left.x * t_l, d_left.y * t_l, d_left.z * t_l);
        let p_right = Vec3::new(
            origin_right.x + d_right_local.x * t_r,
            origin_right.y + d_right_local.y * t_r,
            origin_right.z + d_right_local.z * t_r,
        );
        // Closest point: midpoint of the two skew-line closest approach points.
        Some(Vec3::new(
            0.5 * (p_left.x + p_right.x),
            0.5 * (p_left.y + p_right.y),
            0.5 * (p_left.z + p_right.z),
        ))
    }
}
