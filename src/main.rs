// src/main.rs

use ndarray::Array2;
use std::error::Error;
use std::time::Instant;

// Use the library part of our crate
use freqmut::{config::AppConfig, data::ExperimentData, inference::{run_iterative_inference, InferenceResult}};

fn main() -> Result<(), Box<dyn Error>> {
    let t_program_start = Instant::now();

    // --- 1. Get timepoints ---
    let timepoints = (0..=112).step_by(8).collect();

    // --- 2. Set up Configuration ---
    let config = AppConfig::from_csv("eco15/simu_3_EvoSimulation_Read_Number.csv", "eco15", timepoints);

    // --- 3. Load and Prepare Data ---
    let data = ExperimentData::new(&config)?;

    // --- 4. Run Core Algorithm ---
    let results = run_iterative_inference(&config, &data)?;

    // --- 5. Save Results ---
    save_results(&config, &results)?;

    println!("\nTotal execution time: {:.2}s", t_program_start.elapsed().as_secs_f32());
    println!("\nAnalysis complete. Final 'muti' matrix (first 10 rows):");
    for row in results.final_muti.rows().into_iter().take(10) {
        println!("{:?}", row.to_vec());
    }

    Ok(())
}

/// Writes the final inference results to CSV files.
fn save_results(config: &AppConfig, results: &InferenceResult) -> Result<(), Box<dyn Error>> {
    println!("\nWriting results...");
    
    write_ndarray_to_csv(
        &format!("{}_freqmut_predicted_mutation_info.csv", config.save_prefix),
        &results.final_muti,
    )?;
    
    write_ndarray_to_csv(
        &format!("{}_freqmut_predicted_mean_fitness.csv", config.save_prefix),
        &results.sb_history,
    )?;

    write_ndarray_to_csv(
        &format!("{}_freqmut_predicted_ancestor_fraction.csv", config.save_prefix),
        &results.anc_history,
    )?;

    println!("Successfully wrote all results.");
    Ok(())
}

/// Generic helper to write any 2D ndarray to a CSV file.
fn write_ndarray_to_csv(filename: &str, array: &Array2<f64>) -> Result<(), Box<dyn Error>> {
    println!("  -> {}", filename);
    let mut writer = csv::Writer::from_path(filename)?;
    for row in array.outer_iter() {
        writer.write_record(row.iter().map(|val| val.to_string()))?;
    }
    writer.flush()?;
    Ok(())
}