// src/data.rs

//! Data loading and representation.

use crate::config::AppConfig;
use ndarray::{Array, Array1, Array2, Axis};
use std::error::Error;

/// Holds all the primary data for the experiment.
pub struct ExperimentData {
    /// Raw read counts for each lineage (row) at each timepoint (column).
    pub reads: Array2<f64>,
    /// The specific timepoints at which measurements were taken.
    pub t_points: Array1<i32>,
    /// Total reads across all lineages for each timepoint.
    pub total_reads_per_timepoint: Array1<f64>,
    /// A fine-grained time vector for accurate integration.
    pub time_fine: Array1<f64>,
    /// Number of lineages.
    pub n_lineages: usize,
    /// Number of measurement timepoints.
    pub n_timepoints: usize,
}

impl ExperimentData {
    /// Loads and preprocesses data based on the application configuration.
    pub fn new(config: &AppConfig) -> Result<Self, Box<dyn Error>> {
        println!("Loading data...");
        
        let reads_raw = read_csv_to_ndarray(&config.read_path, false)?;
        
        // Clamp reads to a minimum of 1.0 to avoid issues with log-likelihood.
        let reads = reads_raw.mapv(|x| x.max(1.0));

        let t_points = config.timepoints.clone();

        // Create a fine-grained time vector for interpolation and integration.
        let tfine_i32: Array1<i32> = (config.bounds.tau_min..=config.bounds.tau_max).collect();
        let time_fine: Array1<f64> = tfine_i32.mapv(|x| x as f64);

        let total_reads_per_timepoint: Array1<f64> = reads.sum_axis(Axis(0));
        let n_lineages = reads.nrows();
        let n_timepoints = t_points.len();

        println!(
            "Data loaded. NLin = {}, {} timepoints.",
            n_lineages, n_timepoints
        );

        Ok(Self {
            reads,
            t_points,
            total_reads_per_timepoint,
            time_fine,
            n_lineages,
            n_timepoints,
        })
    }
}


/// Helper function to read a CSV file into an ndarray::Array2<f64>.
fn read_csv_to_ndarray(filepath: &str, has_headers: bool) -> Result<Array2<f64>, Box<dyn Error>> {
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(has_headers)
        .from_path(filepath)?;

    let mut records = Vec::new();
    let mut ncols = 0;
    for result in rdr.records() {
        let record = result?;
        let row: Vec<f64> = record.iter().map(|s| s.parse().unwrap_or(0.0)).collect();
        if ncols == 0 {
            ncols = row.len();
        }
        records.extend_from_slice(&row);
    }
    
    if ncols == 0 {
        return Ok(Array::from_shape_vec((0, 0), records)?);
    }
    let nrows = records.len() / ncols;
    Ok(Array::from_shape_vec((nrows, ncols), records)?)
}