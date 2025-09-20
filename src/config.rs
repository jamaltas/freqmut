// src/config.rs

//! Application configuration and model parameter bounds.

use crate::models::parameters::ModelParameterBounds;

use ndarray::{Array1};

pub struct AppConfig {
    pub read_path: String,
    pub save_prefix: String,
    pub timepoints: Array1<i32>,
    pub non_ecology_penalty: f64,
    pub ecology_penalty: f64,
    pub max_iterations: u64,
    pub convergence_threshold: f64,
    pub bounds: ModelParameterBounds,
}

impl AppConfig {
    pub fn from_csv(read_path: &str, save_prefix: &str, timepoints: Vec<i32>) -> Self {
        Self {
            read_path: read_path.to_string(),
            save_prefix: save_prefix.to_string(),
            timepoints: timepoints.into_iter().collect(),
            non_ecology_penalty: 5.0,
            ecology_penalty: 25.0,
            max_iterations: 45,
            convergence_threshold: 5e-4,
            bounds: ModelParameterBounds {
                s_min: 0.01,
                s_max: 0.25,
                tau_min: -20,
                tau_max: 112,
                xc_min: 0.5,
                xc_max: 0.9,
                m_min: 0.1,
                m_max: 4.5,
                r0_rel_min: 0.5,
                r0_rel_max: 1.5,
            },
        }
    }
}