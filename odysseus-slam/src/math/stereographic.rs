//! Stereographic projection for minimal parameterization of unit vectors
//!
//! This implements stereographic projection following Basalt's approach, which maps
//! 3D unit vectors (directions on a sphere) to 2D points on a plane. This provides
//! a compact, singularity-free representation for bearing vectors.
//!
//! The projection is defined everywhere except at the south pole (0, 0, -1).
//! For forward-pointing rays (typical in cameras), the projection stays small
//! in magnitude near (0, 0), which is numerically well-conditioned.
//!
//! # Projection Formula
//!
//! π(x,y,z) = [x/(d+z), y/(d+z)] where d = √(x²+y²+z²)
//!
//! # Unprojection Formula
//!
//! π⁻¹(u,v) = 2/(1+u²+v²) * [u, v, 1] - [0, 0, 1]

use odysseus_solver::math3d::Vec3;

/// Project a 3D direction vector to 2D using stereographic projection
///
/// Maps a 3D point (typically a unit direction) to a 2D point on the plane.
/// The input does not need to be normalized; the function handles the normalization.
///
/// # Arguments
/// * `p` - 3D point/direction to project
///
/// # Returns
/// 2D stereographic projection [u, v]
pub fn project(p: Vec3<f64>) -> (f64, f64) {
    let norm = (p.x * p.x + p.y * p.y + p.z * p.z).sqrt();
    let denom = norm + p.z;
    let u = p.x / denom;
    let v = p.y / denom;
    (u, v)
}

/// Unproject a 2D point to a 3D unit direction using inverse stereographic projection
///
/// Maps a 2D point back to a 3D unit direction on the sphere.
///
/// # Arguments
/// * `u` - First component of 2D projection
/// * `v` - Second component of 2D projection
///
/// # Returns
/// 3D unit direction vector
pub fn unproject(u: f64, v: f64) -> Vec3<f64> {
    let u2 = u * u;
    let v2 = v * v;
    let r2 = u2 + v2;
    
    let norm_inv = 2.0 / (1.0 + r2);
    
    Vec3::new(
        u * norm_inv,
        v * norm_inv,
        norm_inv - 1.0,
    )
}

/// Project a 3D direction to 2D using stereographic projection (generic version for autodiff)
///
/// This version works with any Real type (f64 or Jet) for automatic differentiation.
///
/// # Arguments
/// * `p` - 3D direction as array [x, y, z]
///
/// # Returns
/// 2D projection as array [u, v]
pub fn project_jet<T, const N: usize>(p: [T; 3]) -> [T; 2]
where
    T: odysseus_solver::Real,
{
    // Compute norm: sqrt(x² + y² + z²)
    let norm_sq = p[0] * p[0] + p[1] * p[1] + p[2] * p[2];
    let norm = norm_sq.sqrt();
    
    // Compute denominator: norm + z
    let denom = norm + p[2];
    
    // Project: [x/(norm+z), y/(norm+z)]
    [p[0] / denom, p[1] / denom]
}

/// Unproject a 2D point to 3D unit direction (generic version for autodiff)
///
/// This version works with any Real type (f64 or Jet) for automatic differentiation.
///
/// # Arguments
/// * `proj` - 2D projection as array [u, v]
///
/// # Returns
/// 3D unit direction as array [x, y, z]
pub fn unproject_jet<T, const N: usize>(proj: [T; 2]) -> [T; 3]
where
    T: odysseus_solver::Real,
{
    let two = T::from_f64(2.0);
    let one = T::from_f64(1.0);
    
    let u = proj[0];
    let v = proj[1];
    
    let u2 = u * u;
    let v2 = v * v;
    let r2 = u2 + v2;
    
    let norm_inv = two / (one + r2);
    
    [
        u * norm_inv,
        v * norm_inv,
        norm_inv - one,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use odysseus_solver::Jet;
    use approx::assert_relative_eq;

    #[test]
    fn test_project_unproject_roundtrip() {
        // Test various directions
        let test_directions = vec![
            Vec3::new(0.0_f64, 0.0, 1.0),     // Forward
            Vec3::new(1.0, 0.0, 1.0),         // Forward-right
            Vec3::new(0.0, 1.0, 1.0),         // Forward-up
            Vec3::new(1.0, 1.0, 1.0),         // Forward-right-up
            Vec3::new(-1.0, -1.0, 1.0),       // Forward-left-down
            Vec3::new(0.5, 0.3, 2.0),         // Arbitrary forward
        ];

        for dir in test_directions {
            // Normalize manually
            let norm: f64 = (dir.x * dir.x + dir.y * dir.y + dir.z * dir.z).sqrt();
            let normalized = Vec3::new(dir.x / norm, dir.y / norm, dir.z / norm);
            
            // Project to 2D
            let (u, v) = project(normalized);
            
            // Unproject back to 3D
            let reconstructed = unproject(u, v);
            
            // Check roundtrip
            assert_relative_eq!(normalized.x, reconstructed.x, epsilon = 1e-10);
            assert_relative_eq!(normalized.y, reconstructed.y, epsilon = 1e-10);
            assert_relative_eq!(normalized.z, reconstructed.z, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_forward_direction_small_projection() {
        // Forward-pointing directions should have small projection values
        let forward = Vec3::new(0.0, 0.0, 1.0);
        let (u, v) = project(forward);
        
        assert_relative_eq!(u, 0.0, epsilon = 1e-10);
        assert_relative_eq!(v, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_jet_project_unproject_roundtrip() {
        // Test with Jets for autodiff
        let x = Jet::<f64, 3>::variable(0.5, 0);
        let y = Jet::<f64, 3>::variable(0.3, 1);
        let z = Jet::<f64, 3>::variable(2.0, 2);
        
        let dir = [x, y, z];
        
        // Normalize
        let norm_sq = dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2];
        let norm = norm_sq.sqrt();
        let normalized = [dir[0] / norm, dir[1] / norm, dir[2] / norm];
        
        // Project
        let proj = project_jet::<Jet<f64, 3>, 3>(normalized);
        
        // Unproject
        let reconstructed = unproject_jet::<Jet<f64, 3>, 3>(proj);
        
        // Check values match
        assert_relative_eq!(normalized[0].value, reconstructed[0].value, epsilon = 1e-10);
        assert_relative_eq!(normalized[1].value, reconstructed[1].value, epsilon = 1e-10);
        assert_relative_eq!(normalized[2].value, reconstructed[2].value, epsilon = 1e-10);
    }

    #[test]
    fn test_unnormalized_input() {
        // Project should work even with unnormalized input
        let unnormalized = Vec3::new(1.0_f64, 2.0, 3.0);
        let (u1, v1) = project(unnormalized);
        
        // Normalize manually
        let norm: f64 = (unnormalized.x * unnormalized.x + unnormalized.y * unnormalized.y + unnormalized.z * unnormalized.z).sqrt();
        let normalized = Vec3::new(unnormalized.x / norm, unnormalized.y / norm, unnormalized.z / norm);
        let (u2, v2) = project(normalized);
        
        assert_relative_eq!(u1, u2, epsilon = 1e-10);
        assert_relative_eq!(v1, v2, epsilon = 1e-10);
    }
}
