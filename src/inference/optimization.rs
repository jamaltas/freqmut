// src/inference/optimization.rs

//! Functions for running optimization for a single lineage.

use super::cost_functions::*;
use super::neldermead::{NelderMead, OptimizationResult as MyOptResult};

// We still use argmin::core for the CostFunction trait, which provides a
// standard interface and error type for our cost functions.
use argmin::core::CostFunction;
use ndarray::{Array, Array1};

/// The results of optimization for a single lineage.
pub struct OptimizationResult {
    /// Converted model parameters: [s, tau, xc, m]. Zero for unused params.
    pub muti_row: Array1<f64>,
    /// Unconstrained model parameters.
    pub cmuti_row: Array1<f64>,
}

/// Creates an initial simplex on the stack for the Nelder-Mead algorithm.
/// This version is compatible with stable Rust by taking DIMS and POINTS as
/// separate const generic parameters.
fn make_simplex_stack<const DIMS: usize, const POINTS: usize>(
    initial_point: &[f64; DIMS],
) -> [[f64; DIMS]; POINTS] {
    // This assertion ensures the core requirement of a simplex is met.
    // In release builds, this check has zero cost.
    assert_eq!(POINTS, DIMS + 1, "Simplex must have DIMS + 1 points.");

    let mut simplex = [[0.0; DIMS]; POINTS];
    simplex[0] = *initial_point;
    for i in 0..DIMS {
        let mut next_point = *initial_point;
        let step = if next_point[i].abs() > 1e-9 { next_point[i] * 0.05 } else { 0.00025 };
        next_point[i] += step;
        simplex[i + 1] = next_point;
    }
    simplex
}

/// For a single lineage, find the best-fitting model by running optimizers.
pub fn find_best_model_for_lineage(ctx: &CostContext) -> OptimizationResult {
    const TOLERANCE: f64 = 1e-6;

    // --- 1. Fit Neutral Model (1 dimension) ---
    // This closure acts as a bridge. It takes a slice `&[f64]` from our solver
    // and creates a temporary `Vec<f64>` to satisfy the `CostFunction` trait's
    // requirement for a `Sized` parameter type. The allocation is tiny and
    // its performance impact is negligible compared to the cost calculation.
    let cost_n = |p: &[f64]| NeutralCost { ctx }.cost(&p.to_vec());
    let simplex_n = make_simplex_stack::<1, 2>(&[-2.0]);
    let mut solver_n = NelderMead::<1, 2>::new(simplex_n);
    let res_n = solver_n.run(&cost_n, 100, TOLERANCE);

    // --- 2. Fit Non-Ecology Model (3 dimensions) ---
    let cost_ne = |p: &[f64]| NonEcologyCost { ctx }.cost(&p.to_vec());
    let initgridne = [
        [-0.01, -0.2, -3.5], [-0.01, -0.2, -2.5], [-0.01, -0.2, -1.5],
        [-0.01, -0.2, -0.5], [-0.01, -0.2, 0.5],
    ];
    let mut best_res_ne: Option<MyOptResult<3>> = None;
    for init_guess in initgridne {
        let simplex = make_simplex_stack::<3, 4>(&init_guess);
        let mut solver = NelderMead::<3, 4>::new(simplex);
        let res = solver.run(&cost_ne, 200, TOLERANCE);
        if best_res_ne.is_none() || res.best_cost < best_res_ne.as_ref().unwrap().best_cost {
            best_res_ne = Some(res);
        }
    }
    let best_res_ne = best_res_ne.unwrap();

    // --- 3. Compare Neutral and Non-Ecology ---
    let mut muti_row = Array1::zeros(4);
    let mut cmuti_row = Array1::zeros(5);

    if res_n.best_cost <= best_res_ne.best_cost + ctx.config.non_ecology_penalty {
        // Neutral model is better or not significantly worse
        return OptimizationResult { muti_row, cmuti_row };
    }

    // --- 4. If Non-Ecology is better, fit Ecology Model (5 dimensions) ---
    let cost_a = |p: &[f64]| EcologyCost { ctx }.cost(&p.to_vec());
    let initgrida = [
        [-0.01, -0.2, -3.5, 0.01, -1.0], [-0.01, -0.2, -2.5, 0.01, -1.0],
        [-0.01, -0.2, -1.5, 0.01, -1.0], [-0.01, -0.2, -0.5, 0.01, -1.0],
        [-0.01, -0.2, 0.5, 0.01, -1.0],
    ];
    let mut best_res_a: Option<MyOptResult<5>> = None;
    for init_guess in initgrida {
        let simplex = make_simplex_stack::<5, 6>(&init_guess);
        let mut solver = NelderMead::<5, 6>::new(simplex);
        let res = solver.run(&cost_a, 300, TOLERANCE);
        if best_res_a.is_none() || res.best_cost < best_res_a.as_ref().unwrap().best_cost {
            best_res_a = Some(res);
        }
    }
    let best_res_a = best_res_a.unwrap();

    // --- 5. Compare Non-Ecology and Ecology ---
    if best_res_a.best_cost + ctx.config.ecology_penalty < best_res_ne.best_cost {
        // Ecology model is significantly better
        let p = best_res_a.best_param;
        muti_row[0] = crate::models::parameters::conv_s(p[1], &ctx.config.bounds);
        muti_row[1] = crate::models::parameters::conv_tau(p[2], &ctx.config.bounds);
        muti_row[2] = crate::models::parameters::conv_xc(p[3], &ctx.config.bounds);
        muti_row[3] = crate::models::parameters::conv_m(p[4], &ctx.config.bounds);
        // Convert the result array `[f64; 5]` to an ndarray slice for assignment
        cmuti_row.assign(&Array::from_vec(p.to_vec()));
    } else {
        // Non-Ecology model is better or not significantly worse
        let p = best_res_ne.best_param;
        muti_row[0] = crate::models::parameters::conv_s(p[1], &ctx.config.bounds);
        muti_row[1] = crate::models::parameters::conv_tau(p[2], &ctx.config.bounds);
        cmuti_row
            .slice_mut(ndarray::s![0..3])
            .assign(&Array::from_vec(p.to_vec()));
    }

    OptimizationResult { muti_row, cmuti_row }
}