//! Inspect EXR file structure to debug loading issues
//!
//! Usage:
//!   cargo run --example exr_inspect -- <path_to_exr>

use exr::prelude::*;
use std::env;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();

    let path = if args.len() >= 2 {
        Path::new(&args[1]).to_path_buf()
    } else {
        eprintln!("Usage: {} <path_to_exr>", args[0]);
        std::process::exit(1);
    };

    println!("Inspecting: {}", path.display());
    println!();

    // Read metadata
    let meta = MetaData::read_from_file(&path, true)?;

    println!("=== EXR Metadata ===");
    println!("Number of layers: {}", meta.headers.len());
    println!();

    for (i, header) in meta.headers.iter().enumerate() {
        println!("Layer {}:", i);
        println!("  Name: {:?}", header.own_attributes.layer_name);
        println!("  Data window: {:?}", header.shared_attributes.display_window);
        println!("  Channels ({}):", header.channels.list.len());
        for (name, channel) in &header.channels.list {
            println!(
                "    - {} (type: {:?}, sampling: {:?})",
                name, channel.sample_type, channel.sampling
            );
        }
        println!();
    }

    // Try to read actual pixel data from first channel
    if let Some(header) = meta.headers.first() {
        if let Some((name, _)) = header.channels.list.first() {
            println!("Attempting to read first channel '{}'...", name);

            // Try reading as flat image with all channels
            let image = read()
                .no_deep_data()
                .largest_resolution_level()
                .all_channels()
                .all_layers()
                .all_attributes()
                .from_file(&path)?;

            println!("Successfully read image!");
            println!("Layer count: {}", image.layer_data.len());

            for (li, layer) in image.layer_data.iter().enumerate() {
                println!("Layer {} channels: {}", li, layer.channel_data.list.len());
                for (ci, channel) in layer.channel_data.list.iter().enumerate() {
                    let sample_count = match &channel.sample_data {
                        FlatSamples::F16(v) => v.len(),
                        FlatSamples::F32(v) => v.len(),
                        FlatSamples::U32(v) => v.len(),
                    };
                    println!("  Channel {}: {} samples, name={}", ci, sample_count, channel.name);
                }
            }
        }
    }

    Ok(())
}
