//! 2D Marginalization Playground
//!
//! CLI tool to run and compare different marginalization approaches.
//!
//! Usage:
//!   cargo run --example 2d_marg_playground -- written-out --viz
//!   cargo run --example 2d_marg_playground -- compare
//!   cargo run --example 2d_marg_playground -- sqrt

mod marginalization_common;

#[path = "2d_marg_written_out.rs"]
mod written_out;

#[path = "2d_marg_matrix.rs"]
mod matrix;

#[path = "2d_marg_sqrt.rs"]
mod sqrt;

use marginalization_common::*;
use clap::{Parser, Subcommand};

#[cfg(feature = "visualization")]
use rerun as rr;

#[derive(Parser)]
#[command(name = "2d_marg_playground")]
#[command(about = "2D Controller Marginalization Playground", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run written-out formulas method
    WrittenOut {
        /// Enable Rerun visualization
        #[arg(short, long)]
        viz: bool,
    },
    /// Run matrix (Schur complement) method
    Matrix {
        /// Enable Rerun visualization
        #[arg(short, long)]
        viz: bool,
    },
    /// Run square root (Cholesky) method
    Sqrt {
        /// Enable Rerun visualization
        #[arg(short, long)]
        viz: bool,
    },
    /// Compare all three methods
    Compare,
}

fn run_method(
    name: &str,
    method_fn: fn(&[Observation]) -> Vec<State>,
    observations: &[Observation],
    true_states: &[State],
    #[allow(unused_variables)] viz: bool,
) {
    println!("🎮 2D Controller Tracking - {}", name);
    println!("{}", "=".repeat(50 + name.len()));
    println!();

    println!("📊 Simulation:");
    println!("  Method: {}", name);
    println!("  Total time steps: {}", N_STEPS);
    println!("  Window size: {}\n", WINDOW_SIZE);

    #[cfg(feature = "visualization")]
    let rec = if viz {
        let rec_name = format!("2d_marg_{}", name.to_lowercase().replace(" ", "_"));
        Some(rr::RecordingStreamBuilder::new(&rec_name).spawn().ok())
    } else {
        None
    };

    println!("🔧 Running optimization...");

    let estimated_trajectory = method_fn(observations);

    #[cfg(feature = "visualization")]
    if let Some(Some(ref rec)) = rec {
        // Simple visualization without marginalized trajectory (just estimated vs truth)
        log_trajectory(rec, 0, true_states, &estimated_trajectory, &Vec::new());
    }

    let (rmse_pos, rmse_vel) = compute_rmse(&estimated_trajectory, true_states);
    println!("\n✅ Complete! RMSE: pos={:.6}, vel={:.6}", rmse_pos, rmse_vel);

    #[cfg(feature = "visualization")]
    if viz {
        println!("\n📺 Open Rerun to see the visualization!");
    }
}

fn compare_all() {
    println!("\n🏁 2D Controller Marginalization - METHOD COMPARISON");
    println!("======================================================\n");

    println!("📊 Test Configuration:");
    println!("  Random seed: 42 (trajectory), 123 (observations)");
    println!("  Time steps: {}", N_STEPS);
    println!("  Window size: {}", WINDOW_SIZE);
    println!("  Position noise σ: {}", POS_NOISE_STD);
    println!("  Velocity noise σ: {}", VEL_NOISE_STD);
    println!("  Motion noise σ: {}\n", MOTION_NOISE_STD);

    // Generate shared test data
    let true_states = simulate_brownian_motion(42);
    let observations = generate_observations(&true_states, 123);

    println!("🔄 Running methods...\n");

    // Method 1: Written-out
    print!("  [1/3] Written-out formulas... ");
    let traj_written = written_out::run(&observations[..]);
    let (rmse_pos_w, rmse_vel_w) = compute_rmse(&traj_written[..], &true_states[..]);
    println!("✓");

    // Method 2: Matrix
    print!("  [2/3] Matrix (Schur complement)... ");
    let traj_matrix = matrix::run(&observations[..]);
    let (rmse_pos_m, rmse_vel_m) = compute_rmse(&traj_matrix[..], &true_states[..]);
    println!("✓");

    // Method 3: Square Root
    print!("  [3/3] Square root (Cholesky)... ");
    let traj_sqrt = sqrt::run(&observations[..]);
    let (rmse_pos_s, rmse_vel_s) = compute_rmse(&traj_sqrt[..], &true_states[..]);
    println!("✓");

    println!("\n📈 Results:");
    println!("┌─────────────────────────────┬──────────────┬──────────────┐");
    println!("│ Method                      │ Pos RMSE     │ Vel RMSE     │");
    println!("├─────────────────────────────┼──────────────┼──────────────┤");
    println!("│ Written-out formulas        │ {:.6}   │ {:.6}   │", rmse_pos_w, rmse_vel_w);
    println!("│ Matrix (Schur complement)   │ {:.6}   │ {:.6}   │", rmse_pos_m, rmse_vel_m);
    println!("│ Square root (Cholesky)      │ {:.6}   │ {:.6}   │", rmse_pos_s, rmse_vel_s);
    println!("└─────────────────────────────┴──────────────┴──────────────┘\n");

    // Find best method
    let methods = [
        ("Written-out", rmse_pos_w, rmse_vel_w),
        ("Matrix", rmse_pos_m, rmse_vel_m),
        ("Square root", rmse_pos_s, rmse_vel_s),
    ];

    let best_pos = methods.iter().min_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).unwrap();
    let best_vel = methods.iter().min_by(|a, b| a.2.partial_cmp(&b.2).unwrap()).unwrap();

    println!("🏆 Winner:");
    println!("  Position accuracy: {} (RMSE = {:.6})", best_pos.0, best_pos.1);
    println!("  Velocity accuracy: {} (RMSE = {:.6})\n", best_vel.0, best_vel.2);

    // Compute relative differences
    println!("📊 Relative Differences (vs best):");
    for (name, rmse_pos, rmse_vel) in &methods {
        let pos_diff_pct = ((rmse_pos - best_pos.1) / best_pos.1) * 100.0;
        let vel_diff_pct = ((rmse_vel - best_vel.2) / best_vel.2) * 100.0;
        println!("  {}: pos +{:.2}%, vel +{:.2}%", name, pos_diff_pct, vel_diff_pct);
    }

    println!("\n✅ Comparison complete!\n");
}

fn main() {
    let cli = Cli::parse();

    // Generate test data (shared across all commands)
    let true_states = simulate_brownian_motion(42);
    let observations = generate_observations(&true_states, 123);

    match cli.command {
        Commands::WrittenOut { viz } => {
            run_method(
                "Written-Out Formulas",
                written_out::run,
                &observations[..],
                &true_states[..],
                viz,
            );
        }
        Commands::Matrix { viz } => {
            run_method(
                "Matrix (Schur Complement)",
                matrix::run,
                &observations[..],
                &true_states[..],
                viz,
            );
        }
        Commands::Sqrt { viz } => {
            run_method(
                "Square Root (Cholesky)",
                sqrt::run,
                &observations[..],
                &true_states[..],
                viz,
            );
        }
        Commands::Compare => {
            compare_all();
        }
    }
}
