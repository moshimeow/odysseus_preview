//! Generate a Kalibr-style AprilGrid PNG (+ metadata JSON) for use as a Blender
//! calibration board texture.
//!
//! # Layout
//!
//! Kalibr/aprilgrid-rs boards are NOT just tags-with-gaps; they are a
//! checkerboard of small black squares ("corner squares") interleaved with
//! AprilTags.  The corner squares are what the saddle-point detector finds —
//! without them, no detection.
//!
//! Pattern is on a (2N+1) × (2M+1) cell grid where (row+col) even cells are
//! filled and (row+col) odd cells are white:
//!
//!   row even: cells are small black squares (size = `tag_size × spacing_ratio`)
//!   row odd : cells are AprilTags (size = `tag_size`)
//!
//! Tag IDs are assigned row-by-row from the BOTTOM of the image (matches
//! `AprilGridLayout`: tag id = row * cols + col, row 0 at bottom).
//!
//! # Tag36H11 bit encoding
//!
//! Each tag is 10×10 cells (2-cell black border + 6×6 data area).  Bit 35 of
//! the u64 code is the top-left data cell, bit 0 is the bottom-right
//! (row-major in image coordinates).  A 1 bit renders white, 0 black.
//!
//! # Metadata
//!
//! A JSON file is written next to the PNG with `pattern_size_m`,
//! `pixels_per_meter`, `pattern_origin_px` (image-space pixel of tag 0's
//! bottom-left corner), and `tag_corner_check` (image-space pixel positions
//! of tag 0's 4 corners — useful for confirming the layout matches the
//! Rust `AprilGridLayout::corner_position`).
//!
//! # Usage
//!
//! ```text
//! cargo run --release --example gen_aprilgrid --features aprilgrid -- \
//!     --rows 6 --cols 6 --tag-size-m 0.030 --spacing-ratio 0.3 \
//!     --cell-px 20 --output blender_stuff/aprilgrid_6x6.png
//! ```

#[cfg(not(feature = "aprilgrid"))]
compile_error!("gen_aprilgrid requires --features aprilgrid");

use aprilgrid::tag_families::T36H11;
use clap::Parser;
use image::{GrayImage, ImageBuffer, Luma};
use serde::Serialize;
use std::path::PathBuf;

const BORDER: u32 = 2;
const EDGE: u32 = 6;
const CELLS_PER_TAG: u32 = BORDER * 2 + EDGE; // 10

#[derive(Parser)]
#[command(name = "gen_aprilgrid", about = "Generate a Kalibr-style AprilGrid PNG + metadata")]
struct Args {
    #[arg(long, default_value_t = 6)]
    rows: u32,

    #[arg(long, default_value_t = 6)]
    cols: u32,

    /// Physical tag side length in metres (used only for metadata)
    #[arg(long, default_value_t = 0.030)]
    tag_size_m: f64,

    /// Gap / corner-square size as a fraction of tag size (Kalibr default 0.3)
    #[arg(long, default_value_t = 0.3)]
    spacing_ratio: f64,

    /// Pixels per tag cell (tag becomes 10×cell_px wide)
    #[arg(long, default_value_t = 20)]
    cell_px: u32,

    /// Outer white margin as a fraction of tag size (e.g. 0.5)
    #[arg(long, default_value_t = 0.5)]
    margin_ratio: f64,

    /// Output PNG path. Metadata JSON is written next to it (.json suffix).
    #[arg(long, default_value = "aprilgrid.png")]
    output: PathBuf,
}

#[derive(Serialize)]
struct Metadata {
    rows: u32,
    cols: u32,
    tag_size_m: f64,
    spacing_ratio: f64,
    /// Side length of the pattern itself (no outer margin), metres.
    pattern_size_m: f64,
    /// PNG dimensions, including outer white margin.
    image_width_px: u32,
    image_height_px: u32,
    /// Pattern (excluding outer margin) dimensions in pixels.
    pattern_width_px: u32,
    pattern_height_px: u32,
    /// PNG pixels per metre — multiply board-frame metres by this to get
    /// pixel offsets into the pattern.
    pixels_per_meter: f64,
    /// Image-space pixel of tag 0's bottom-left corner (matches
    /// `AprilGridLayout::corner_position(tag_id=0, corner_idx=0)` at (0,0,0)).
    tag0_bl_image_px: [f64; 2],
    /// All 4 corners of tag 0 in image-space pixels [BL, BR, TR, TL].
    tag0_corners_image_px: [[f64; 2]; 4],
}

/// Pixel value for one cell of a tag36h11 tag.
fn tag_cell_value(code: u64, cell_row: u32, cell_col: u32) -> u8 {
    if cell_row < BORDER
        || cell_row >= BORDER + EDGE
        || cell_col < BORDER
        || cell_col >= BORDER + EDGE
    {
        return 0;
    }
    let data_row = cell_row - BORDER;
    let data_col = cell_col - BORDER;
    // bit 35 = top-left data cell, bit 0 = bottom-right (row-major).
    let bit_idx = (EDGE * EDGE - 1) - (data_row * EDGE + data_col);
    if (code >> bit_idx) & 1 == 1 { 255 } else { 0 }
}

/// Paint a flat black square.
fn paint_square(img: &mut GrayImage, x0: u32, y0: u32, size: u32) {
    for dy in 0..size {
        for dx in 0..size {
            img.put_pixel(x0 + dx, y0 + dy, Luma([0u8]));
        }
    }
}

/// Paint a single tag.
fn paint_tag(img: &mut GrayImage, x0: u32, y0: u32, code: u64, cell_px: u32) {
    for cell_row in 0..CELLS_PER_TAG {
        for cell_col in 0..CELLS_PER_TAG {
            let value = tag_cell_value(code, cell_row, cell_col);
            let px_x0 = x0 + cell_col * cell_px;
            let px_y0 = y0 + cell_row * cell_px;
            for dy in 0..cell_px {
                for dx in 0..cell_px {
                    img.put_pixel(px_x0 + dx, px_y0 + dy, Luma([value]));
                }
            }
        }
    }
}

fn main() {
    let args = Args::parse();

    let tag_px = CELLS_PER_TAG * args.cell_px;
    // Small corner-square size in pixels. Equal to the per-tag spacing.
    let small_px = (args.spacing_ratio * tag_px as f64).round() as u32;
    let margin_px = (args.margin_ratio * tag_px as f64).round() as u32;

    // Pattern footprint: (N+1) small squares + N tags per side.
    let pattern_w = (args.cols + 1) * small_px + args.cols * tag_px;
    let pattern_h = (args.rows + 1) * small_px + args.rows * tag_px;

    let img_w = pattern_w + 2 * margin_px;
    let img_h = pattern_h + 2 * margin_px;

    let mut img: GrayImage = ImageBuffer::from_pixel(img_w, img_h, Luma([255u8]));

    // Pattern origin in pixels (top-left of pattern in image space).
    let origin_x = margin_px;
    let origin_y = margin_px;

    // Render the (2*rows+1) × (2*cols+1) cell grid.  In image space, image-row
    // 0 is at the TOP, but our tag-id convention has tag-row 0 at the BOTTOM.
    // So image_row r corresponds to tag-grid row (2*rows - r).
    let cell_rows = 2 * args.rows + 1;
    let cell_cols = 2 * args.cols + 1;

    // Helper: image-space x of the LEFT edge of cell `col`.
    // Even cols are small squares, odd cols are tags.
    let cell_x = |col: u32| -> u32 {
        let small_count = (col + 1) / 2;
        let tag_count = col / 2;
        origin_x + small_count * small_px + tag_count * tag_px
    };
    let cell_y = |image_row: u32| -> u32 {
        let small_count = (image_row + 1) / 2;
        let tag_count = image_row / 2;
        origin_y + small_count * small_px + tag_count * tag_px
    };

    for image_row in 0..cell_rows {
        for col in 0..cell_cols {
            // Only "filled" cells where (image_row + col) is even.
            if (image_row + col) % 2 != 0 {
                continue;
            }
            let x0 = cell_x(col);
            let y0 = cell_y(image_row);
            if image_row % 2 == 0 {
                // Small black corner square.
                paint_square(&mut img, x0, y0, small_px);
            } else {
                // AprilTag at tag-grid position.  image_row = 1, 3, 5, ...
                // corresponds to tag-row 5, 4, 3, ... (with rows=6).
                let tag_grid_row = args.rows - 1 - (image_row - 1) / 2;
                let tag_grid_col = (col - 1) / 2;
                let tag_id = (tag_grid_row * args.cols + tag_grid_col) as usize;
                if tag_id >= T36H11.len() {
                    continue;
                }
                paint_tag(&mut img, x0, y0, T36H11[tag_id], args.cell_px);
            }
        }
    }

    img.save(&args.output)
        .unwrap_or_else(|e| panic!("Failed to save {:?}: {e}", args.output));

    // Metadata.
    let pixels_per_meter = (tag_px as f64) / args.tag_size_m;
    let pattern_size_m = (pattern_w as f64) / pixels_per_meter;

    // Tag 0 (bottom-left) lives at image_row = 2*rows - 1, col = 1.
    let tag0_image_row = 2 * args.rows - 1;
    let tag0_x0 = cell_x(1) as f64;
    let tag0_y_top = cell_y(tag0_image_row) as f64;
    let tag0_y_bottom = tag0_y_top + tag_px as f64;
    let bl = [tag0_x0, tag0_y_bottom];
    let br = [tag0_x0 + tag_px as f64, tag0_y_bottom];
    let tr = [tag0_x0 + tag_px as f64, tag0_y_top];
    let tl = [tag0_x0, tag0_y_top];

    let meta = Metadata {
        rows: args.rows,
        cols: args.cols,
        tag_size_m: args.tag_size_m,
        spacing_ratio: args.spacing_ratio,
        pattern_size_m,
        image_width_px: img_w,
        image_height_px: img_h,
        pattern_width_px: pattern_w,
        pattern_height_px: pattern_h,
        pixels_per_meter,
        tag0_bl_image_px: bl,
        tag0_corners_image_px: [bl, br, tr, tl],
    };

    let meta_path = args.output.with_extension("json");
    let meta_str = serde_json::to_string_pretty(&meta).expect("serialize metadata");
    std::fs::write(&meta_path, meta_str)
        .unwrap_or_else(|e| panic!("Failed to write metadata {:?}: {e}", meta_path));

    println!(
        "Wrote {}×{} px PNG ({} × {} tags) to {:?}",
        img_w, img_h, args.rows, args.cols, args.output
    );
    println!(
        "  Tag: {} px   Small square: {} px   Margin: {} px",
        tag_px, small_px, margin_px
    );
    println!(
        "  Pattern: {}×{} px = {:.4}×{:.4} m   ({:.2} px/m)",
        pattern_w, pattern_h, pattern_size_m, pattern_size_m, pixels_per_meter
    );
    println!("  Metadata: {:?}", meta_path);
}
