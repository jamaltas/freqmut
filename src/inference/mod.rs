// src/inference/mod.rs

//! Core iterative inference algorithm.

mod cost_functions;
mod optimization;

use self::cost_functions::CostContext;
use self::optimization::{find_best_model_for_lineage, OptimizationResult};

use crate::config::AppConfig;
use crate::data::ExperimentData;
use crate::interp_zero_alloc::{interp_zero_alloc, InterpMode};
use crate::models::parameters::*;

use ndarray::{s, Array1, Array2};
use rayon::prelude::*;
use std::error::Error;
use quadrature::integrate;

/// Holds the state that is updated at each iteration of the algorithm.
struct InferenceState {
    /// Mean fitness over time.
    sbi: Array1<f64>,
    /// Fraction of ancestral genotype over time.
    ancf: Array1<f64>,
    /// Converted parameters for each lineage's best-fit model.
    muti: Array2<f64>,
    /// Unconstrained parameters for each lineage's best-fit model.
    cmuti: Array2<f64>,
}

/// Contains the final results of the inference process.
pub struct InferenceResult {
    /// Final converted parameters for each lineage.
    pub final_muti: Array2<f64>,
    /// History of mean fitness (sbi) over all iterations.
    pub sb_history: Array2<f64>,
    /// History of ancestor fraction (ancf) over all iterations.
    pub anc_history: Array2<f64>,
}

/// Runs the main Expectation-Maximization-like iterative inference algorithm.
pub fn run_iterative_inference(
    config: &AppConfig,
    data: &ExperimentData,
) -> Result<InferenceResult, Box<dyn Error>> {
    let mut state = initialize_state(data, config);

    let mut sb_history_flat = Vec::new();
    let mut anc_history_flat = Vec::new();
    let mut actual_iterations = 0;

    for iter in 1..=config.max_iterations {
        let t_iter_start = std::time::Instant::now();
        println!("Iteration: {}", iter);

        // --- E-Step: Infer lineage parameters given global sbi and ancf ---
        let iter_inputs = prepare_iteration_inputs(&state.sbi, &state.ancf, data, config);
        
        let optimization_results: Vec<OptimizationResult> = (0..data.n_lineages)
            .into_par_iter()
            //.into_iter()
            .map(|i| {
                let ctx = CostContext {
                    lineage_idx: i,
                    config,
                    data,
                    ee_matrix: &iter_inputs.ee_matrix,
                    afi: &iter_inputs.afi,
                    ancf: &state.ancf,
                    ancf_integral_t: &iter_inputs.ancf_integral_t,
                    ancf_integral_v: &iter_inputs.ancf_integral_v,
                    ancf_time_lookup_y: &iter_inputs.ancf_time_lookup_y,
                    ancf_time_lookup_x: &iter_inputs.ancf_time_lookup_x,
                };
                find_best_model_for_lineage(&ctx)
            })
            .collect();
        
        for (i, result) in optimization_results.into_iter().enumerate() {
            state.muti.row_mut(i).assign(&result.muti_row);
            state.cmuti.row_mut(i).assign(&result.cmuti_row);
        }

        // --- M-Step: Update global sbi and ancf given lineage parameters ---
        let (new_sbi, new_ancf) = update_global_state(&state, &iter_inputs, data, config);

        // --- Convergence Check ---
        let diff = &new_sbi - &state.sbi;
        let sum_sq_diff = diff.mapv(|x| x.powi(2)).sum();
        println!("  Convergence metric (Sum Sq Diff): {:.4e}", sum_sq_diff);

        // --- Store history and update state for next iteration ---
        sb_history_flat.extend_from_slice(new_sbi.as_slice().unwrap());
        anc_history_flat.extend_from_slice(new_ancf.as_slice().unwrap());
        actual_iterations += 1;
        
        state.sbi = new_sbi;
        state.ancf = new_ancf;

        if sum_sq_diff < config.convergence_threshold && iter > 1 {
            println!("  Convergence threshold reached. Stopping iteration.");
            break;
        }

        println!("Time: {:.2}s", t_iter_start.elapsed().as_secs_f32());
    }

    let kmfine = data.time_fine.len();
    let sb_history = Array2::from_shape_vec((actual_iterations, kmfine), sb_history_flat)?;
    let anc_history = Array2::from_shape_vec((actual_iterations, kmfine), anc_history_flat)?;

    Ok(InferenceResult {
        final_muti: state.muti,
        sb_history,
        anc_history,
    })
}


/// Intermediate data calculated once per iteration.
struct IterationInputs<'a> {
    /// Ancestor fraction integrated, exp(-integral(sbi)).
    afi: Array1<f64>,
    /// Effective ancestor fraction matrix for neutral growth.
    ee_matrix: Array2<f64>,
    /// Time points for the ancf integral lookup table.
    ancf_integral_t: Vec<f64>,
    /// Values for the ancf integral lookup table.
    ancf_integral_v: Vec<f64>,
    /// Y-values (ancf) for reverse time lookup.
    ancf_time_lookup_y: Vec<f64>,
    /// X-values (time) for reverse time lookup.
    ancf_time_lookup_x: Vec<f64>,
    _phantom: std::marker::PhantomData<&'a ()>,
}


/// Pre-calculates interpolation tables and other data needed for the optimization step.
fn prepare_iteration_inputs<'a>(
    sbi: &Array1<f64>,
    ancf: &Array1<f64>,
    data: &ExperimentData,
    config: &AppConfig,
) -> IterationInputs<'a> {
    let interp_mode = InterpMode::Extrapolate;

    // Calculate integral of s_b to get ancestor fraction (afi)
    let ci: Array1<f64> = (config.bounds.tau_min..=config.bounds.tau_max)
        .map(|i| {
            let integrand = |u: f64| interp_zero_alloc(data.time_fine.as_slice().unwrap(), sbi.as_slice().unwrap(), u, &interp_mode);
            integrate(integrand, 0.0, i as f64, 1e-6).integral
        })
        .collect();
    let afi = ci.mapv(|x| (-x).exp());

    // Calculate EE matrix
    let afi_slice = afi.slice(s![(0-config.bounds.tau_min) as usize..=(config.bounds.tau_max-config.bounds.tau_min) as usize]);
    let mut ee_matrix = Array2::zeros((config.bounds.tau_max as usize + 1, config.bounds.tau_max as usize + 1));
    for r_idx in 0..=config.bounds.tau_max as usize {
        for c_idx in 0..=config.bounds.tau_max as usize {
            ee_matrix[[r_idx, c_idx]] = afi_slice[c_idx] / afi_slice[r_idx];
        }
    }

    // Prepare interpolation tables for the ecology model
    let mut ancfs_monotonic = ancf.clone();
    for i in 1..ancfs_monotonic.len() {
        ancfs_monotonic[i] = ancfs_monotonic[i].min(ancfs_monotonic[i-1]);
    }
    let axc: Vec<bool> = ancfs_monotonic.iter().map(|&v| v >= 0.9 * config.bounds.xc_min && v <= 1.1 * config.bounds.xc_max).collect();
    let ancf_time_lookup_y: Vec<f64> = ancfs_monotonic.iter().zip(&axc).filter(|&(_, &ax)| ax).map(|(&v, _)| v).rev().collect();
    let ancf_time_lookup_x: Vec<f64> = data.time_fine.iter().zip(&axc).filter(|&(_, &ax)| ax).map(|(&v, _)| v).rev().collect();
    let tamax_int = interp_zero_alloc(&ancf_time_lookup_y, &ancf_time_lookup_x, config.bounds.xc_min, &interp_mode).ceil() as i32;

    let ancf_integral_t: Vec<f64> = (config.bounds.tau_min..=tamax_int).map(|i| i as f64).collect();
    let ancf_integral_v: Vec<f64> = ancf_integral_t.iter().map(|&i| {
        let integrand = |u: f64| interp_zero_alloc(data.time_fine.as_slice().unwrap(), ancf.as_slice().unwrap(), u, &interp_mode);
        integrate(integrand, config.bounds.tau_min as f64, i, 1e-6).integral
    }).collect();

    IterationInputs {
        afi,
        ee_matrix,
        ancf_integral_t,
        ancf_integral_v,
        ancf_time_lookup_y,
        ancf_time_lookup_x,
        _phantom: std::marker::PhantomData,
    }
}

/// Recalculates the global `sbi` and `ancf` arrays based on the newly inferred lineage parameters.
fn update_global_state(
    state: &InferenceState,
    iter_inputs: &IterationInputs,
    data: &ExperimentData,
    config: &AppConfig,
) -> (Array1<f64>, Array1<f64>) {
    let mut nsbi = Array1::zeros(data.time_fine.len());
    let mut nancf = Array1::zeros(data.time_fine.len());
    let interp_mode = InterpMode::Extrapolate;

    for i in 0..data.n_lineages {
        if state.muti.row(i).sum() == 0.0 { continue; }

        let cmuti_i = state.cmuti.row(i);
        let r_ref = data.reads[[i, 0]] / data.total_reads_per_timepoint[0];
        let x0 = conv_r0(cmuti_i[0], r_ref, &config.bounds);
        let s_val = conv_s(cmuti_i[1], &config.bounds);
        let tau = conv_tau(cmuti_i[2], &config.bounds);
        let eef_t1 = interp_zero_alloc(
            data.time_fine.as_slice().unwrap(), iter_inputs.afi.as_slice().unwrap(),
            data.t_points[0] as f64, &interp_mode
        );
        
        let (mcurve, scurve) = if state.muti[[i, 2]] > 0.0 { // Ecology model was chosen
            let xc = conv_xc(cmuti_i[3], &config.bounds);
            let m = conv_m(cmuti_i[4], &config.bounds);
            let a_tau = interp_zero_alloc(data.time_fine.as_slice().unwrap(), state.ancf.as_slice().unwrap(), tau, &interp_mode);
            
            let mc = data.time_fine.mapv(|tk| {
                if tk < tau { return 0.0; }
                let mut rt = 0.01 * x0 * interp_zero_alloc(data.time_fine.as_slice().unwrap(), iter_inputs.afi.as_slice().unwrap(), tk, &interp_mode) / eef_t1 * (s_val * (tk - tau)).exp();
                if a_tau > xc {
                    let tend = if interp_zero_alloc(data.time_fine.as_slice().unwrap(), state.ancf.as_slice().unwrap(), tk, &interp_mode) < xc {
                        interp_zero_alloc(&iter_inputs.ancf_time_lookup_y, &iter_inputs.ancf_time_lookup_x, xc, &interp_mode)
                    } else { tk };
                    let aif_tend = interp_zero_alloc(&iter_inputs.ancf_integral_t, &iter_inputs.ancf_integral_v, tend, &interp_mode);
                    let aif_tau = interp_zero_alloc(&iter_inputs.ancf_integral_t, &iter_inputs.ancf_integral_v, tau, &interp_mode);
                    rt *= (m * (aif_tend - aif_tau - xc * (tend - tau))).exp();
                }
                rt
            });
            let sc = data.time_fine.mapv(|tt| {
                let a_tt = interp_zero_alloc(data.time_fine.as_slice().unwrap(), state.ancf.as_slice().unwrap(), tt, &interp_mode);
                s_val + if a_tt > xc { m * (a_tt - xc) } else { 0.0 }
            });
            (mc, sc)

        } else { // Non-ecology model
            let mc = data.time_fine.mapv(|tk| if tk < tau { 0.0 } else { 0.01 * x0 * interp_zero_alloc(data.time_fine.as_slice().unwrap(), iter_inputs.afi.as_slice().unwrap(), tk, &interp_mode) / eef_t1 * (s_val * (tk - tau)).exp() });
            (mc, Array1::from_elem(data.time_fine.len(), s_val))
        };

        nancf += &mcurve;
        nsbi += &(&mcurve * &scurve);
    }

    let nancf_max_1 = nancf.mapv(|x| x.max(1.0));
    let new_sbi = &nsbi / &nancf_max_1;
    let new_ancf = 1.0 - &nancf / &nancf_max_1;
    (new_sbi, new_ancf)
}


/// Computes an initial guess for the mean fitness `sbi`.
fn initialize_state(data: &ExperimentData, config: &AppConfig) -> InferenceState {
    let reads_mean_col0 = data.reads.slice(s![.., 0]).mean().unwrap_or(0.0);
    let putneut: Vec<bool> = data.reads.outer_iter().map(|row| {
        row[0] < reads_mean_col0 && row[0] > row[row.len() - 1] && row[row.len() - 1] > 1.0
    }).collect();

    let mut sbic = Array1::zeros(data.n_timepoints);
    for k in 1..data.n_timepoints {
        let mut vals_k: Vec<f64> = data.reads.outer_iter().zip(&putneut).filter(|&(_, p)| *p).map(|(r, _)| r[k]).collect();
        let mut vals_k_minus_1: Vec<f64> = data.reads.outer_iter().zip(&putneut).filter(|&(_, p)| *p).map(|(r, _)| r[k-1]).collect();
        
        let median_k = if vals_k.is_empty() { 1.0 } else { vals_k.sort_by(|a, b| a.partial_cmp(b).unwrap()); vals_k[vals_k.len() / 2] };
        let median_k_minus_1 = if vals_k_minus_1.is_empty() { 1.0 } else { vals_k_minus_1.sort_by(|a,b|a.partial_cmp(b).unwrap()); vals_k_minus_1[vals_k_minus_1.len() / 2] };

        let val = -(median_k / median_k_minus_1).ln() / ((data.t_points[k] - data.t_points[k - 1]) as f64);
        sbic[k] = val.max(sbic[k - 1]);
    }

    let mut sb_t = vec![config.bounds.tau_min as f64];
    sb_t.extend(data.t_points.iter().map(|&x| x as f64));
    let mut sb_val = vec![0.0];
    sb_val.extend(sbic.to_vec());
    
    let interp_mode = InterpMode::Extrapolate;
    let sbi = data.time_fine.mapv(|t| interp_zero_alloc(&sb_t, &sb_val, t, &interp_mode));
    
    // Initial ancf is based on the initial sbi
    let ci: Array1<f64> = (config.bounds.tau_min..=config.bounds.tau_max)
        .map(|i| {
            let integrand = |u: f64| interp_zero_alloc(data.time_fine.as_slice().unwrap(), sbi.as_slice().unwrap(), u, &interp_mode);
            integrate(integrand, 0.0, i as f64, 1e-6).integral
        })
        .collect();
    let afi = ci.mapv(|x| (-x).exp());
    let ancf = data.time_fine.mapv(|t| interp_zero_alloc(data.time_fine.as_slice().unwrap(), afi.as_slice().unwrap(), t, &interp_mode));

    InferenceState {
        sbi,
        ancf,
        muti: Array2::zeros((data.n_lineages, 4)),
        cmuti: Array2::zeros((data.n_lineages, 5)),
    }
}
