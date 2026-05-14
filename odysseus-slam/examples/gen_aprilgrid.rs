//! Generate an AprilGrid PNG for use as a Blender calibration board texture.
//!
//! Renders a tag36h11 AprilGrid from the same bit codes the detector uses, so
//! what gets rendered is guaranteed to be detectable by the same crate.
//!
//! # Tag36H11 encoding
//!
//! Each tag is 10×10 cells: a 2-cell-wide black border surrounding a 6×6
//! data area. The u64 code for tag `id` has bit 35 = top-left data cell and
//! bit 0 = bottom-right data cell (row-major, top-to-bottom). A 1 bit renders
//! white; a 0 bit renders black.
//!
//! # Usage
//!
//! ```text
//! cargo run --release --example gen_aprilgrid --features aprilgrid -- \
//!     --rows 6 --cols 6 --cell-px 80 --spacing-ratio 0.3 \
//!     --output blender_stuff/aprilgrid_6x6.png
//! ```

#[cfg(not(feature = "aprilgrid"))]
compile_error!("gen_aprilgrid requires --features aprilgrid");

use aprilgrid::tag_families::T36H11;
use clap::Parser;
use image::{GrayImage, ImageBuffer, Luma};

const BORDER: u32 = 2;
const EDGE: u32 = 6;
const CELLS_PER_TAG: u32 = BORDER * 2 + EDGE; // 10

#[derive(Parser)]
#[command(name = "gen_aprilgrid", about = "Generate an AprilGrid PNG texture")]
struct Args {
    /// Number of tag rows
    #[arg(long, default_value_t = 6)]
    rows: u32,

    /// Number of tag columns
    #[arg(long, default_value_t = 6)]
    cols: u32,

    /// Pixels per tag cell (tag will be 10×cell_px wide)
    #[arg(long, default_value_t = 80)]
    cell_px: u32,

    /// Gap between tags as a fraction of tag size (e.g. 0.3)
    #[arg(long, default_value_t = 0.3)]
    spacing_ratio: f64,

    /// Outer white margin as a fraction of tag size (e.g. 0.5)
    #[arg(long, default_value_t = 0.5)]
    margin_ratio: f64,

    /// Output PNG path
    #[arg(long, default_value = "aprilgrid.png")]
    output: String,
}

/// Return the grayscale pixel value for one cell of a tag36h11 tag.
fn tag_cell_value(code: u64, cell_row: u32, cell_col: u32) -> u8 {
    // Border region (2 cells wide) is always black.
    if cell_row < BORDER
        || cell_row >= BORDER + EDGE
        || cell_col < BORDER
        || cell_col >= BORDER + EDGE
    {
        return 0;
    }
    let data_row = cell_row - BORDER;
    let data_col = cell_col - BORDER;
    // bit 35 = top-left (data_row=0, data_col=0), bit 0 = bottom-right.
    let bit_idx = (EDGE * EDGE - 1) - (data_row * EDGE + data_col);
    if (code >> bit_idx) & 1 == 1 { 255 } else { 0 }
}

fn main() {
    let args = Args::parse();

    let tag_px = CELLS_PER_TAG * args.cell_px;
    let gap_px = (args.spacing_ratio * tag_px as f64).round() as u32;
    let margin_px = (args.margin_ratio * tag_px as f64).round() as u32;

    let img_w = margin_px * 2 + args.cols * tag_px + (args.cols - 1) * gap_px;
    let img_h = margin_px * 2 + args.rows * tag_px + (args.rows - 1) * gap_px;

    let mut img: GrayImage = ImageBuffer::from_pixel(img_w, img_h, Luma([255u8]));

    // Tag id = row * cols + col.  Row 0 is at the TOP of the image
    // (image-row convention; the bottom of the image corresponds to low Y in
    // board space, i.e. tag row 0 should appear at the bottom).
    // We render row 0 at top here and rely on Blender's V-flip (OpenGL UV
    // origin is bottom-left, so the stored PNG is displayed right-side up).
    for tag_row in 0..args.rows {
        for tag_col in 0..args.cols {
            let tag_id = (tag_row * args.cols + tag_col) as usize;
            if tag_id >= T36H11.len() {
                eprintln!(
                    "WARNING: tag_id {} exceeds T36H11 table ({} entries). Stopping.",
                    tag_id,
                    T36H11.len()
                );
                break;
            }
            let code = T36H11[tag_id];

            let origin_x = margin_px + tag_col * (tag_px + gap_px);
            let origin_y = margin_px + tag_row * (tag_px + gap_px);

            for cell_row in 0..CELLS_PER_TAG {
                for cell_col in 0..CELLS_PER_TAG {
                    let value = tag_cell_value(code, cell_row, cell_col);
                    let px_x0 = origin_x + cell_col * args.cell_px;
                    let px_y0 = origin_y + cell_row * args.cell_px;
                    for dy in 0..args.cell_px {
                        for dx in 0..args.cell_px {
                            img.put_pixel(px_x0 + dx, px_y0 + dy, Luma([value]));
                        }
                    }
                }
            }
        }
    }

    img.save(&args.output)
        .unwrap_or_else(|e| panic!("Failed to save {}: {e}", args.output));

    println!(
        "Wrote {}×{} px AprilGrid ({} × {} tags) to {}",
        img_w, img_h, args.rows, args.cols, args.output
    );
    println!(
        "  Tag size: {}×{} px   Gap: {} px   Margin: {} px",
        tag_px, tag_px, gap_px, margin_px
    );
}
