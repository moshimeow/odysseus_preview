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

use odysseus_solver::math3d::Vec3;
use odysseus_solver::Real;

/// Common interface for intrinsic camera models.
///
/// Implementors expose a single forward projection that's compatible with
/// `Real`-valued scalars (so the calibration optimizer can run autodiff
/// through the projection).
///
/// `NUM_INTRINSICS` reports how many scalar parameters the model carries — it
/// drives parameter-block sizing in the calibration solver (Phase 3).
pub trait CameraModel<T: Real>: Sized + Copy {
    const NUM_INTRINSICS: usize;

    fn project(&self, point_cam: Vec3<T>) -> (T, T);
}
