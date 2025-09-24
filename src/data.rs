// src/data.rs
use ndarray::{Array1, Array2, Axis};
use std::error::Error;
use csv;

/// Loads a CSV file into an ndarray Array2<f64>.
pub fn load_csv(path: &str) -> Result<Array2<f64>, Box<dyn Error>> {
    let mut reader = csv::ReaderBuilder::new().has_headers(false).from_path(path)?;
    let mut data = Vec::new();
    let mut n_cols = 0;
    for result in reader.records() {
        let record = result?;
        if n_cols == 0 {
            n_cols = record.len();
        }
        for field in record.iter() {
            data.push(field.trim().parse::<f64>()?);
        }
    }
    let n_rows = data.len() / n_cols;
    Ok(Array2::from_shape_vec((n_rows, n_cols), data)?)
}

/// Preprocesses the raw simulation data.
/// This corresponds to the initial data handling section of the Julia script.
pub fn preprocess_data(
    prefix: &str,
) -> (Array2<f64>, Array1<f64>) {
    
    let reads_path = format!("{}/simu_3_EvoSimulation_Read_Number.csv", prefix);
    //let reads_path = format!("{}/levy_50000.csv", prefix);
    let mut reads = load_csv(&reads_path).expect("Failed to load Read Number CSV");

    // Total read depth per time point
    let r: Array1<f64> = reads.sum_axis(Axis(0));

    // Replace read counts of 0 with 1
    reads.mapv_inplace(|x| if x < 1.0 { 1.0 } else { x });
    
    (reads, r)
}