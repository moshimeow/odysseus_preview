//! Camera models
//!
//! Provides several intrinsic projection models, all generic over the scalar
//! type `T: Real` so they work with both `f64` and `Jet<f64, N>` for autodiff:
//!
//! - [`PinholeCamera`] — standard pinhole.
//! - [`KannalaBrandtCamera`] — KB4 fisheye (used by ORB-SLAM3).
//! - [`DoubleSphereCamera`] — fisheye with closed-form unprojection
//!   (Usenko et al. 2018).
//! - [`BlenderPolyFisheyeCamera`] — 4th-degree polynomial fisheye matching
//!   Blender Cycles' "Fisheye Polynomial" sensor.
//! - [`StereoCamera`] — pair of pinholes with a horizontal baseline.
//!
//! All models that implement [`CameraModel`] can be used generically by the
//! intrinsic calibration optimizer.

pub mod blender_fisheye;
pub mod double_sphere;
pub mod kannala_brandt;
pub mod pinhole;
pub mod stereo;

pub use blender_fisheye::BlenderPolyFisheyeCamera;
pub use double_sphere::DoubleSphereCamera;
pub use kannala_brandt::KannalaBrandtCamera;
pub use pinhole::PinholeCamera;
pub use stereo::StereoCamera;

// Re-exported here so downstream code can write `use crate::{...}`
// and pick up both the model trait and the jet-lifting trait in one shot.

use odysseus_solver::math3d::Vec3;
use odysseus_solver::{Jet, Real};

/// Common interface for intrinsic camera models.
///
/// Implementors expose forward projection and unprojection compatible with
/// `Real`-valued scalars (so the calibration optimizer and the controller
/// tracker can both run autodiff through the camera).
///
/// `NUM_INTRINSICS` reports how many scalar parameters the model carries.
pub trait CameraModel<T: Real>: Sized + Copy {
    const NUM_INTRINSICS: usize;

    /// Project a 3D point in camera frame to a 2D pixel.
    fn project(&self, point_cam: Vec3<T>) -> (T, T);

    /// Unproject a 2D pixel to a 3D ray in camera frame.  Convention: the
    /// returned ray has positive Z but is NOT necessarily unit-length.
    /// Multiply the ray by `depth / ray.z` to scale to a specific Z, or by
    /// `1.0 / |ray|` to get a unit vector.
    fn unproject(&self, u: T, v: T) -> Vec3<T>;

    /// Unproject a pixel to a 3D point at the given camera-frame depth (Z).
    /// Default impl scales the unit-ish ray; concrete models may override.
    fn unproject_at_depth(&self, u: T, v: T, depth: T) -> Vec3<T> {
        let ray = self.unproject(u, v);
        let scale = depth / ray.z;
        Vec3::new(ray.x * scale, ray.y * scale, depth)
    }
}

/// Convert an `f64`-valued camera to a `Jet<f64, N>` camera with constant
/// intrinsics (zero derivatives).  Used when the tracker treats intrinsics as
/// fixed but still needs to autodiff through projection w.r.t. the 3D point.
///
/// Each f64-valued model implements this; callers do:
/// ```ignore
/// let cam_jet: PinholeCamera<Jet<f64, N>> = pinhole_f64.constant_jet::<N>();
/// let (u, v) = cam_jet.project(point_jet);
/// ```
pub trait CameraConstantJet: CameraModel<f64> {
    /// Lifted camera type carrying `Jet<f64, N>` intrinsics.
    type Jet<const N: usize>: CameraModel<Jet<f64, N>>;

    fn constant_jet<const N: usize>(&self) -> Self::Jet<N>;
}

pub mod calibration;
