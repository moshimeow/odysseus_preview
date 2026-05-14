//! Quick diagnostic: run the raw TagDetector on an image and print what it finds.
//! Usage: cargo run --release --example detect_test --features aprilgrid -- path/to/image.png
#[cfg(not(feature = "aprilgrid"))]
compile_error!("requires --features aprilgrid");

use aprilgrid::detector::TagDetector;
use aprilgrid::TagFamily;

fn main() {
    let path = std::env::args().nth(1).expect("usage: detect_test <image>");
    let img = image::open(&path).expect("failed to open image");
    println!("Image {}x{}", img.width(), img.height());
    let det = TagDetector::new(&TagFamily::T36H11, None);
    let res = det.detect(&img);
    println!("Tags found: {}", res.len());
    let mut ids: Vec<u32> = res.keys().copied().collect();
    ids.sort();
    for id in ids {
        println!("  tag {id}: {:?}", res[&id]);
    }
}
