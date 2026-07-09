//! Calibrate a Double Sphere camera from AprilGrid images.
//!
//! Loads images produced by `synth_calib_resample` (or real camera images
//! with the same metadata format), runs the AprilGrid detector on each,
//! then calls `calibrate_double_sphere`.
//!
//! When the metadata contains `ground_truth_ds`, the recovered intrinsics are
//! compared to the ground truth and the per-parameter error is printed.
//!
//! # Usage
//!
//! ```text
//! calibrate_ds --input blender_stuff/calib_ds/metadata.json \
//!              --fx 280 --fy 280 --cx 320 --cy 240 --xi -0.2 --alpha 0.58
//! ```
//!
//! If no initial intrinsics are given, the ground truth from the metadata is
//! used (convenient for validating synthetic data).  For real data supply
//! `--fx`, `--fy`, `--cx`, `--cy` and reasonable `--xi`/`--alpha` priors
//! (e.g. xi=-0.2 alpha=0.5 works for most fisheye lenses).

use clap::Parser;
use odysseus_camera::calibration::detector::{AprilGridDetector, TagFamily};
use odysseus_camera::calibration::{estimate_board_pose, AprilGridLayout, BoardObservation};
use odysseus_solver::math3d::{SE3, SO3};
use odysseus_camera::calibration::intrinsics::{calibrate_double_sphere, DoubleSphereCalibConfig};
use odysseus_camera::DoubleSphereCamera;
use odysseus_solver::math3d::{Mat3, Vec3};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

// ── CLI ───────────────────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(
    name = "calibrate_ds",
    about = "Calibrate a Double Sphere camera from AprilGrid images"
)]
struct Args {
    /// Path to metadata.json (from synth_calib_resample or real capture)
    #[arg(long)]
    input: PathBuf,

    /// Initial fx (pixels). Defaults to ground_truth_ds.fx if present.
    #[arg(long)]
    fx: Option<f64>,
    #[arg(long)]
    fy: Option<f64>,
    #[arg(long)]
    cx: Option<f64>,
    #[arg(long)]
    cy: Option<f64>,
    /// Initial xi (sphere separation). Typical prior: -0.2
    #[arg(long)]
    xi: Option<f64>,
    /// Initial alpha (blend factor). Typical prior: 0.5
    #[arg(long)]
    alpha: Option<f64>,

    /// Max solver iterations
    #[arg(long, default_value_t = 100)]
    max_iter: usize,

    /// Huber loss threshold in pixels
    #[arg(long, default_value_t = 1.5)]
    huber_delta: f64,

    /// Print per-iteration solver output
    #[arg(long, default_value_t = false)]
    verbose: bool,

    /// Write the recovered camera to JSON.  Schema:
    /// `{"model":"double_sphere", "fx":..., "fy":..., "cx":..., "cy":...,
    ///   "xi":..., "alpha":..., "image_width":..., "image_height":...,
    ///   "rms_pixel_error":...}`
    /// Loadable by `controller_tracking::load_ds_calibration`.
    #[arg(long)]
    write_json: Option<PathBuf>,
}

// ── JSON types ────────────────────────────────────────────────────────────────

#[derive(Deserialize)]
struct Metadata {
    ground_truth_ds: Option<GroundTruthDs>,
    april_grid: AprilGridJson,
    frames: Vec<FrameEntry>,
    /// Image dimensions for real captures. Synthetic metadata stores these
    /// inside `ground_truth_ds`; real metadata sets them at the top level.
    #[serde(default)]
    image_width: Option<u32>,
    #[serde(default)]
    image_height: Option<u32>,
}

#[derive(Deserialize)]
struct GroundTruthDs {
    fx: f64,
    fy: f64,
    cx: f64,
    cy: f64,
    xi: f64,
    alpha: f64,
    #[serde(default)]
    image_width: u32,
    #[serde(default)]
    image_height: u32,
}

#[derive(Deserialize)]
struct AprilGridJson {
    tag_rows: u32,
    tag_cols: u32,
    tag_size_m: f64,
    tag_spacing_ratio: f64,
}

#[derive(Deserialize)]
struct FrameEntry {
    frame_idx: usize,
    filename: String,
    /// Optional initial guess. If omitted, computed from the detected
    /// AprilGrid corners via the planar-pose bootstrapper.
    #[serde(default)]
    board_t_camera: Option<[[f64; 4]; 4]>,
}

// ── SE3 from 4×4 row-major matrix ────────────────────────────────────────────

fn se3_from_mat4(m: &[[f64; 4]; 4]) -> SE3<f64> {
    let rot = Mat3 {
        x_axis: Vec3::new(m[0][0], m[1][0], m[2][0]),
        y_axis: Vec3::new(m[0][1], m[1][1], m[2][1]),
        z_axis: Vec3::new(m[0][2], m[1][2], m[2][2]),
    };
    let t = Vec3::new(m[0][3], m[1][3], m[2][3]);
    SE3::from_rotation_translation(SO3::from_matrix(rot), t)
}

// ── Main ──────────────────────────────────────────────────────────────────────

fn main() {
    let args = Args::parse();

    let meta_str = std::fs::read_to_string(&args.input)
        .unwrap_or_else(|e| panic!("Failed to read {}: {e}", args.input.display()));
    let meta: Metadata =
        serde_json::from_str(&meta_str).unwrap_or_else(|e| panic!("Failed to parse metadata: {e}"));

    // Build initial camera from CLI or ground truth fallback.
    let gt = meta.ground_truth_ds.as_ref();
    let init_camera = DoubleSphereCamera::new(
        args.fx
            .or(gt.map(|g| g.fx))
            .expect("--fx required (no ground_truth_ds in metadata)"),
        args.fy.or(gt.map(|g| g.fy)).expect("--fy required"),
        args.cx.or(gt.map(|g| g.cx)).expect("--cx required"),
        args.cy.or(gt.map(|g| g.cy)).expect("--cy required"),
        args.xi.or(gt.map(|g| g.xi)).expect("--xi required"),
        args.alpha
            .or(gt.map(|g| g.alpha))
            .expect("--alpha required"),
    );

    println!(
        "Initial camera:  fx={:.3} fy={:.3} cx={:.3} cy={:.3} xi={:.4} alpha={:.4}",
        init_camera.fx,
        init_camera.fy,
        init_camera.cx,
        init_camera.cy,
        init_camera.xi,
        init_camera.alpha
    );

    // Build detector.
    let layout = AprilGridLayout::new(
        meta.april_grid.tag_rows,
        meta.april_grid.tag_cols,
        meta.april_grid.tag_size_m,
        meta.april_grid.tag_spacing_ratio,
    );
    let detector = AprilGridDetector::new(layout, TagFamily::T36H11);
    let image_dir = args.input.parent().unwrap_or_else(|| Path::new("."));

    // Detect corners and collect initial extrinsics.
    let mut observations: Vec<BoardObservation> = Vec::new();
    let mut initial_extrinsics: HashMap<usize, SE3<f64>> = HashMap::new();
    let mut total_corners = 0usize;

    for frame in &meta.frames {
        let img_path = image_dir.join(&frame.filename);
        let img = image::open(&img_path)
            .unwrap_or_else(|e| panic!("Failed to open {}: {e}", img_path.display()));

        let obs = detector.detect(&img, frame.frame_idx);
        let n = obs.corners.len();
        println!("  frame {:03}: {} corners detected", frame.frame_idx, n);

        if n >= 4 {
            let pose = match &frame.board_t_camera {
                Some(m) => se3_from_mat4(m),
                None => match estimate_board_pose(&init_camera, &obs) {
                    Ok(p) => p,
                    Err(e) => {
                        println!("    (skipped — planar-pose bootstrap failed: {:?})", e);
                        continue;
                    }
                },
            };
            total_corners += n;
            initial_extrinsics.insert(frame.frame_idx, pose);
            observations.push(obs);
        } else {
            println!("    (skipped — too few corners)");
        }
    }

    println!(
        "\nTotal corners across {} frames: {}",
        observations.len(),
        total_corners
    );
    if observations.is_empty() {
        eprintln!("ERROR: no usable frames. Check that the images are correct.");
        std::process::exit(1);
    }

    // Run calibration.
    let config = DoubleSphereCalibConfig {
        max_iterations: args.max_iter,
        huber_delta_pixels: args.huber_delta,
        verbose: args.verbose,
        ..Default::default()
    };

    println!("\nRunning calibration...");
    let result = calibrate_double_sphere(init_camera, &initial_extrinsics, &observations, &config);

    let cam = &result.camera;
    println!(
        "\nRecovered camera: fx={:.3} fy={:.3} cx={:.3} cy={:.3} xi={:.4} alpha={:.4}",
        cam.fx, cam.fy, cam.cx, cam.cy, cam.xi, cam.alpha
    );
    println!(
        "RMS reprojection error: {:.4} px  ({} iterations)",
        result.final_rms_pixel_error, result.iterations
    );

    // Optionally write recovered camera to JSON.
    if let Some(out_path) = &args.write_json {
        #[derive(Serialize)]
        struct DsCalibrationJson {
            model: &'static str,
            fx: f64,
            fy: f64,
            cx: f64,
            cy: f64,
            xi: f64,
            alpha: f64,
            image_width: u32,
            image_height: u32,
            rms_pixel_error: f64,
        }
        let json = DsCalibrationJson {
            model: "double_sphere",
            fx: cam.fx,
            fy: cam.fy,
            cx: cam.cx,
            cy: cam.cy,
            xi: cam.xi,
            alpha: cam.alpha,
            image_width: meta
                .image_width
                .or_else(|| meta.ground_truth_ds.as_ref().map(|g| g.image_width))
                .unwrap_or(0),
            image_height: meta
                .image_height
                .or_else(|| meta.ground_truth_ds.as_ref().map(|g| g.image_height))
                .unwrap_or(0),
            rms_pixel_error: result.final_rms_pixel_error,
        };
        let s = serde_json::to_string_pretty(&json).expect("serialize");
        std::fs::write(out_path, s)
            .unwrap_or_else(|e| panic!("Failed to write {}: {e}", out_path.display()));
        println!("Wrote calibration JSON: {}", out_path.display());
    }

    // Compare to ground truth if available.
    if let Some(gt) = &meta.ground_truth_ds {
        println!(
            "\nGround truth:     fx={:.3} fy={:.3} cx={:.3} cy={:.3} xi={:.4} alpha={:.4}",
            gt.fx, gt.fy, gt.cx, gt.cy, gt.xi, gt.alpha
        );
        println!(
            "Errors:           Δfx={:.4} Δfy={:.4} Δcx={:.4} Δcy={:.4} Δxi={:.5} Δalpha={:.5}",
            cam.fx - gt.fx,
            cam.fy - gt.fy,
            cam.cx - gt.cx,
            cam.cy - gt.cy,
            cam.xi - gt.xi,
            cam.alpha - gt.alpha
        );
    }
}
