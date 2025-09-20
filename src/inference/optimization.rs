// src/inference/optimization.rs

//! Functions for running optimization for a single lineage.

use super::cost_functions::*;
use argmin::core::{Executor, IterState};
use argmin::solver::neldermead::NelderMead;
use ndarray::{array, Array, Array1};

/// The results of optimization for a single lineage.
pub struct OptimizationResult {
    /// Converted model parameters: [s, tau, xc, m]. Zero for unused params.
    pub muti_row: Array1<f64>,
    /// Unconstrained model parameters.
    pub cmuti_row: Array1<f64>,
}

/// Creates an initial simplex for the Nelder-Mead algorithm from a single starting point.
fn make_simplex(initial_point: &[f64]) -> Vec<Vec<f64>> {
    let n = initial_point.len();
    let mut simplex = Vec::with_capacity(n + 1);
    simplex.push(initial_point.to_vec());
    for i in 0..n {
        let mut next_point = initial_point.to_vec();
        let step = if next_point[i].abs() > 1e-9 { next_point[i] * 0.05 } else { 0.00025 };
        next_point[i] += step;
        simplex.push(next_point);
    }
    simplex
}

/// For a single lineage, find the best-fitting model (Neutral, Non-Ecology, or Ecology)
/// by running optimizers and comparing costs with penalties.
pub fn find_best_model_for_lineage(ctx: &CostContext) -> OptimizationResult {
    // --- 1. Fit Neutral Model ---
    let simplex_n = make_simplex(&[-2.0]);
    let solver_n = NelderMead::new(simplex_n);
    let res_n = Executor::new(NeutralCost { ctx }, solver_n)
        .configure(|s| s.max_iters(100))
        .run()
        .expect("Neutral model optimization failed");

    // --- 2. Fit Non-Ecology Model ---
    let initgridne = array![
        [-0.01, -0.2, -3.5], [-0.01, -0.2, -2.5], [-0.01, -0.2, -1.5],
        [-0.01, -0.2, -0.5], [-0.01, -0.2, 0.5]
    ];
    let mut best_res_ne: Option<IterState<Vec<f64>, (), (), (), (), f64>> = None;
    for init_guess in initgridne.outer_iter() {
        let simplex = make_simplex(init_guess.as_slice().unwrap());
        let solver = NelderMead::new(simplex);
        let res = Executor::new(NonEcologyCost { ctx }, solver)
            .configure(|s| s.max_iters(200))
            .run()
            .expect("Non-Ecology optimization failed");
        if best_res_ne.is_none() || res.state.best_cost < best_res_ne.as_ref().unwrap().best_cost {
            best_res_ne = Some(res.state.clone());
        }
    }
    let best_res_ne = best_res_ne.unwrap();

    // --- 3. Compare Neutral and Non-Ecology ---
    let mut muti_row = Array1::zeros(4);
    let mut cmuti_row = Array1::zeros(5);

    if res_n.state().best_cost <= best_res_ne.best_cost + ctx.config.non_ecology_penalty {
        // Neutral model is better or not significantly worse
        return OptimizationResult { muti_row, cmuti_row };
    }

    // --- 4. If Non-Ecology is better, fit Ecology Model ---
    let initgrida = array![
        [-0.01, -0.2, -3.5, 0.01, -1.0], [-0.01, -0.2, -2.5, 0.01, -1.0],
        [-0.01, -0.2, -1.5, 0.01, -1.0], [-0.01, -0.2, -0.5, 0.01, -1.0],
        [-0.01, -0.2, 0.5, 0.01, -1.0]
    ];
    let mut best_res_a: Option<IterState<Vec<f64>, (), (), (), (), f64>> = None;
    for init_guess in initgrida.outer_iter() {
        let simplex = make_simplex(init_guess.as_slice().unwrap());
        let solver = NelderMead::new(simplex);
        let res = Executor::new(EcologyCost { ctx }, solver)
            .configure(|s| s.max_iters(300))
            .run()
            .expect("Ecology optimization failed");
        if best_res_a.is_none() || res.state.best_cost < best_res_a.as_ref().unwrap().best_cost {
            best_res_a = Some(res.state.clone());
        }
    }
    let best_res_a = best_res_a.unwrap();

    // --- 5. Compare Non-Ecology and Ecology ---
    if best_res_a.best_cost + ctx.config.ecology_penalty < best_res_ne.best_cost {
        // Ecology model is significantly better
        let p = best_res_a.best_param.unwrap();
        muti_row[0] = crate::models::parameters::conv_s(p[1], &ctx.config.bounds);
        muti_row[1] = crate::models::parameters::conv_tau(p[2], &ctx.config.bounds);
        muti_row[2] = crate::models::parameters::conv_xc(p[3], &ctx.config.bounds);
        muti_row[3] = crate::models::parameters::conv_m(p[4], &ctx.config.bounds);
        cmuti_row.assign(&Array::from(p));
    } else {
        // Non-Ecology model is better or not significantly worse
        let p = best_res_ne.best_param.unwrap();
        muti_row[0] = crate::models::parameters::conv_s(p[1], &ctx.config.bounds);
        muti_row[1] = crate::models::parameters::conv_tau(p[2], &ctx.config.bounds);
        cmuti_row.slice_mut(ndarray::s![0..3]).assign(&Array::from(p));
    }

    OptimizationResult { muti_row, cmuti_row }
}