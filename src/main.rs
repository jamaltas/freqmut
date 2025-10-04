// src/main.rs
mod data;
mod utils;
mod models;
mod neldermead;

use ndarray::{arr2, s, Array1, Array2, ArrayView1, Axis};
use rayon::prelude::*;
use std::time::Instant;
use argmin::core::{CostFunction};
use crate::utils::{median, LinearInterpolator, StatefulInterpolator};
use crate::models::*;
use crate::neldermead::*;
use std::error::Error;
use csv;

use quad_gk::{quad_gk,GKConfig};
use std::sync::Arc;

use egobox_doe::{Lhs, LhsKind, SamplingMethod};

// --- Global constants --- (no changes)
const PREFIX: &str = "levy_exp";
const TAUMIN: i32 = -48;
const TAUMAX: i32 = 112;
const G: usize = 8;
const INITFRAC: f64 = 0.02;
const NEPENALTY: f64 = 30.0;
const APENALTY: f64 = 0.0;
const KAP: f64 = 2.5;
const NITER: usize = 30;

// Parameter ranges (no changes)
const XC_MIN: f64 = 0.3;
const XC_MAX: f64 = 0.9;

// CHANGED: The struct now holds both unconverted and converted parameters for clarity.
struct BarcodeFitResult {
    // Log priors (negative log likelihood)
    log_prior_neutral: f64,
    log_prior_no_eco: f64,
    log_prior_eco: f64,
    
    // Unconverted parameters (raw from optimizer)
    params_neutral: [f64; 1],
    params_no_eco: [f64; 3],
    params_eco: [f64; 5],

    // Converted parameters (physically meaningful values)
    params_neutral_conv: [f64; 1],
    params_no_eco_conv: [f64; 3],
    params_eco_conv: [f64; 5],

    // For the existing logic that drives the next iteration
    muti_row: Array1<f64>,
    cmuti_row: Array1<f64>,
}


/// Writes a 2D ndarray of f64 to a CSV file. (no changes)
fn write_ndarray_to_csv(data: &Array2<f64>, path: &str) -> Result<(), Box<dyn Error>> {
    let mut writer = csv::Writer::from_path(path)?;
    for row in data.rows() {
        writer.write_record(row.iter().map(|&val| val.to_string()))?;
    }
    writer.flush()?;
    Ok(())
}

/// Creates an initial simplex as a fixed-size array for the Nelder-Mead algorithm. (no changes)
fn make_simplex<const DIMS: usize, const POINTS: usize>(
    initial_point: &[f64; DIMS]
) -> [[f64; DIMS]; POINTS] {
    assert_eq!(POINTS, DIMS + 1, "The number of simplex points must be DIMS + 1.");
    let a: f64 = 0.00025; let b: f64 = 0.05; let mut simplex = [[0.0; DIMS]; POINTS];
    simplex[0] = *initial_point;
    for i in 0..DIMS {
        let mut next_point = *initial_point;
        next_point[i] = initial_point[i] * (1.0 + b) + a;
        simplex[i + 1] = next_point;
    }
    simplex
}

fn main() -> Result<(), Box<dyn Error>> {
    // 1. Load and preprocess data (no changes)
    println!("Loading and preprocessing data...");
    let (reads_raw, r_raw) = data::preprocess_data(PREFIX);
    let nlin = reads_raw.nrows();
    let reads = Arc::new(reads_raw); let r = Arc::new(r_raw);

    // 2. Setup time grids and initial parameters (no changes)
    //let t_vec: Vec<i32> = (0..=TAUMAX).step_by(G).collect();
    let t_vec = vec![0, 16, 32, 40, 48, 64, 72, 80, 88, 96, 104, 112];
    let km = t_vec.len();
    let tfine_vec: Vec<i32> = (TAUMIN..=TAUMAX).collect();
    let kmfine = tfine_vec.len();
    let tfine_f64_vec: Vec<f64> = tfine_vec.iter().map(|&x| x as f64).collect();
    let t = Arc::new(t_vec); let tfine = Arc::new(tfine_vec); let tfine_f64 = Arc::new(tfine_f64_vec);
    let xlimits_ne = arr2(&[ [-1.5, 1.5], [-1.5, 1.5], [-1.5, 1.5], ]);
    let initgridne = Lhs::new(&xlimits_ne).kind(LhsKind::Centered).sample(150);
    let xlimits_a = arr2(&[ [-1.5, 1.5], [-1.5, 1.5], [-1.5, 1.5], [-1.5, 1.5], [-1.5, 1.5], ]);
    let initgrida = Lhs::new(&xlimits_a).kind(LhsKind::Centered).sample(150);

    // 3. Initial guess for mean fitness (sb) (no changes)
    let first_col_mean = reads.column(0).mean().unwrap();
    let putneut_indices: Vec<usize> = (0..nlin) .filter(|&i| { let first = reads[[i, 0]]; let last  = reads[[i, km - 1]]; first < first_col_mean && first > last && last > 1.0 }).collect();
    let mut sbic = Array1::zeros(km);
    for k in 1..km {
        let mut col_k_vals: Vec<f64> = putneut_indices.iter().map(|&i| reads[[i, k]] / r[k]).collect();
        let mut col_km1_vals: Vec<f64> = putneut_indices.iter().map(|&i| reads[[i, k - 1]] / r[k - 1]).collect();
        let median_k = median(&mut col_k_vals); let median_km1 = median(&mut col_km1_vals);
        let ratio = median_k / median_km1; let dt = (t[k] as f64) - (t[k - 1] as f64);
        sbic[k] = (-ratio.ln() / dt).max(sbic[k - 1]);
    }
    let mut sb_t_pts: Vec<f64> = Vec::with_capacity(1 + t.len()); sb_t_pts.push(TAUMIN as f64); sb_t_pts.extend(t.iter().map(|&x| x as f64));
    let mut sb_y_pts: Vec<f64> = Vec::with_capacity(1 + sbic.len()); sb_y_pts.push(0.0); sb_y_pts.extend(sbic.iter().cloned());
    let mut sb = LinearInterpolator::new(sb_t_pts, sb_y_pts);
    
    // 4. Main iteration loop
    let mut sbmat = Array2::zeros((NITER, kmfine));
    let mut ancmat = Array2::zeros((NITER, kmfine));
    let mut muti = Array2::zeros((nlin, 4));

    // NEW: Expanded matrix to hold unconverted and converted parameters for all 3 fits.
    let mut fit_details_mat = Array2::zeros((nlin, 3 + 7 + 11)); // 21 columns total

    for iter in 0..NITER {
        let start_time = Instant::now();
        println!("Iteration: {}", iter + 1);

        // --- Update population-level variables for this iteration (no changes) ---
        let sb_arc = Arc::new(sb.clone());
        let ci: Vec<f64> = (TAUMIN..=TAUMAX).map(|i| { 
            let sb_clone = sb_arc.clone(); 
            let integrand = Arc::new(move |u: f64| sb_clone.eval(u));
            let range = 0.0..i as f64;
            quad_gk!(integrand, range, rel_tol = 1.5e-8).value 
        }).collect();
        let afi: Array1<f64> = Array1::from_vec(ci.iter().map(|&c| (-c).exp()).collect());
        let ancf = if iter == 0 { afi.clone() } else { ancmat.row(iter - 1).to_owned() };
        let a = Arc::new(LinearInterpolator::new(tfine_f64.to_vec(), ancf.to_vec()));
        let mut ancfs_mono = ancf.clone(); for i in 1..ancf.len() { ancfs_mono[i] = ancfs_mono[i].min(ancfs_mono[i - 1]); }
        let mut ta_x = Vec::new(); let mut ta_y = Vec::new();
        for i in 0..kmfine { if ancfs_mono[i] >= 0.9*XC_MIN && ancfs_mono[i] <= 1.1*XC_MAX { ta_x.push(ancfs_mono[i]); ta_y.push(tfine_f64[i]); } }
        ta_x.reverse(); ta_y.reverse();
        let ta = Arc::new(LinearInterpolator::new(ta_x, ta_y));
        let tamax = ta.eval(XC_MIN).ceil() as i32;
        let ai_vec: Vec<f64> = (TAUMIN..=tamax).map(|i| { 
            let a_clone = a.clone(); 
            let a_integrand = Arc::new(move |u: f64| a_clone.eval(u)); 
            let range = TAUMIN as f64..i as f64;
            quad_gk!(a_integrand, range, rel_tol = 1.5e-8).value
        }).collect();
        let aif = Arc::new(LinearInterpolator::new((TAUMIN..=tamax).map(|i| i as f64).collect(), ai_vec));
        let eef = Arc::new(LinearInterpolator::new(tfine_f64.to_vec(), afi.to_vec()));
        let v: Array1<f64> = afi.slice(s![(0 - TAUMIN) as usize..]).to_owned();
        let ee = Arc::new(v.mapv(|x| 1.0 / x).insert_axis(Axis(1)) * v.insert_axis(Axis(0)));

        // --- Parallel fitting for each lineage ---
        let results: Vec<BarcodeFitResult> = (0..nlin)
        .into_par_iter()
        .map(|i| {
            let (reads, r, t, _, _, eef, a, ta, aif, ee) = ( reads.clone(), r.clone(), t.clone(), tfine.clone(), tfine_f64.clone(), eef.clone(), a.clone(), ta.clone(), aif.clone(), ee.clone() );

            // --- 1. Fit Neutral Model ---
            let cost_n_fn = NeutralCost { i, reads: reads.clone(), r: r.clone(), t: t.clone(), km, ee: ee.clone(), kap: KAP};
            let cost_n = |p: &[f64; 1]| cost_n_fn.cost(&p.to_vec()); let simplex_n = make_simplex(&[-2.0]);
            let mut solver_n = NelderMead::<1, 2>::new(simplex_n).with_alpha(1.0).unwrap().with_gamma(2.5).unwrap().with_rho(0.4).unwrap().with_sigma(0.4).unwrap();
            let res_n = solver_n.run(&cost_n, 250, 1e-8);
            
            // --- 2. Fit Non-Ecology Model ---
            let cost_ne_fn = NonEcologyCost { i, reads: reads.clone(), r: r.clone(), t: t.clone(), km, eef: eef.clone(), kap: KAP, initfrac: INITFRAC, ee: ee.clone(), taumin: TAUMIN, taumax: TAUMAX };
            let mut best_res_ne: Option<OptimizationResult<3>> = None;
            for guess in initgridne.rows() {
                let mut guess_array = [0.0; 3]; guess.iter().enumerate().for_each(|(idx, &val)| guess_array[idx] = val);
                let cost_ne = |p: &[f64; 3]| cost_ne_fn.cost(&p.to_vec()); let simplex = make_simplex(&guess_array);
                let mut solver = NelderMead::<3, 4>::new(simplex).with_alpha(1.0).unwrap().with_gamma(2.5).unwrap().with_rho(0.4).unwrap().with_sigma(0.4).unwrap();
                let res = solver.run(&cost_ne, 1000, 1e-8);
                if best_res_ne.is_none() || res.best_cost < best_res_ne.as_ref().unwrap().best_cost { best_res_ne = Some(res); }
            }
            let best_res_ne = best_res_ne.unwrap();

    
            // --- 3. Fit Ecology Model ---
            let cost_a_fn = EcologyCost { i, reads: reads.clone(), r: r.clone(), t: t.clone(), km, eef: eef.clone(), a: a.clone(), ta: ta.clone(), aif: aif.clone(), kap: KAP, initfrac: INITFRAC, ee: ee.clone(), taumin: TAUMIN, taumax: TAUMAX };
            let mut best_res_a: Option<OptimizationResult<5>> = None;
            for guess in initgrida.rows() {
                let mut guess_array = [0.0; 5]; guess.iter().enumerate().for_each(|(idx, &val)| guess_array[idx] = val);
                let cost_a = |p: &[f64; 5]| cost_a_fn.cost(&p.to_vec()); let simplex = make_simplex(&guess_array);
                let mut solver = NelderMead::<5,6>::new(simplex).with_alpha(1.0).unwrap().with_gamma(2.5).unwrap().with_rho(0.4).unwrap().with_sigma(0.4).unwrap();
                let res = solver.run(&cost_a, 1000, 1e-8);
                if best_res_a.is_none() || res.best_cost < best_res_a.as_ref().unwrap().best_cost { best_res_a = Some(res); }
            }
            let best_res_a = best_res_a.unwrap();
            

            // NEW: Store raw and converted parameters for all models
            let params_n = res_n.best_param;
            let params_ne = best_res_ne.best_param;
            let params_a = best_res_a.best_param;

            let params_n_conv = [ conv_r0(params_n[0], r[0], reads[[i, 0]] / r[0], KAP) ];
            let params_ne_conv = [ conv_r0(params_ne[0], r[0], reads[[i, 0]] / r[0], KAP), conv_s(params_ne[1]), conv_tau(params_ne[2], TAUMIN, TAUMAX) ];
            let params_a_conv = [ conv_r0(params_a[0], r[0], reads[[i, 0]] / r[0], KAP), conv_s(params_a[1]), conv_tau(params_a[2], TAUMIN, TAUMAX), conv_xc(params_a[3]), conv_m(params_a[4]) ];

            // --- 4. Compare models to determine parameters for the next iteration (logic unchanged) ---
            let mut muti_row = Array1::zeros(4);
            let mut cmuti_row = Array1::zeros(5);
            
            
            if res_n.best_cost > best_res_ne.best_cost + NEPENALTY {
                if best_res_a.best_cost + APENALTY < best_res_ne.best_cost {
                    muti_row[0] = params_a_conv[1]; // s
                    muti_row[1] = params_a_conv[2]; // tau
                    muti_row[2] = params_a_conv[3]; // xc
                    muti_row[3] = params_a_conv[4]; // m
                    cmuti_row.assign(&ArrayView1::from(&params_a));
                } else {
                    muti_row[0] = params_ne_conv[1]; // s
                    muti_row[1] = params_ne_conv[2]; // tau
                    cmuti_row.slice_mut(s![0..3]).assign(&ArrayView1::from(&params_ne));
                }
            }
            
            // --- 5. Return all results for storage ---
            BarcodeFitResult {
                log_prior_neutral: res_n.best_cost,
                log_prior_no_eco: best_res_ne.best_cost,
                log_prior_eco: best_res_a.best_cost,
                params_neutral: params_n,
                params_no_eco: params_ne,
                params_eco: params_a,
                params_neutral_conv: params_n_conv,
                params_no_eco_conv: params_ne_conv,
                params_eco_conv: params_a_conv,
                muti_row,
                cmuti_row,
            }
        }).collect();

        // --- Update sbi and ancf for next iteration (and store detailed results) ---
        let mut cmuti = Array2::zeros((nlin, 5));
        for (i, result) in results.into_iter().enumerate() {
            muti.row_mut(i).assign(&result.muti_row);
            cmuti.row_mut(i).assign(&result.cmuti_row);

            // NEW: Populate the expanded details matrix
            let mut detail_row = fit_details_mat.row_mut(i);
            // Neutral model results
            detail_row[0] = result.log_prior_neutral;
            detail_row.slice_mut(s![1..2]).assign(&ArrayView1::from(&result.params_neutral));      // Unconverted
            detail_row.slice_mut(s![2..3]).assign(&ArrayView1::from(&result.params_neutral_conv)); // Converted
            // No-Ecology model results
            detail_row[3] = result.log_prior_no_eco;
            detail_row.slice_mut(s![4..7]).assign(&ArrayView1::from(&result.params_no_eco));      // Unconverted
            detail_row.slice_mut(s![7..10]).assign(&ArrayView1::from(&result.params_no_eco_conv)); // Converted
            // Ecology model results
            detail_row[10] = result.log_prior_eco;
            detail_row.slice_mut(s![11..16]).assign(&ArrayView1::from(&result.params_eco));      // Unconverted
            detail_row.slice_mut(s![16..21]).assign(&ArrayView1::from(&result.params_eco_conv)); // Converted
        }

        // --- This block is unchanged ---
        let mut nsbi = Array1::zeros(kmfine); let mut nancf = Array1::zeros(kmfine);
        for i in 0..nlin {
            if muti.row(i).iter().any(|&x| x != 0.0) {
                let mcurve; let mut scurve = Array1::zeros(kmfine);
                if muti[[i, 2]] != 0.0 || muti[[i, 3]] != 0.0 { // Ecology
                    mcurve = mcurve_a(cmuti.row(i).as_slice().unwrap(), i, reads.clone(), r.clone(), tfine.clone(), eef.clone(), a.clone(), ta.clone(), aif.clone(), INITFRAC, KAP, TAUMIN, TAUMAX, t[0]);
                    let mut a_stateful = StatefulInterpolator::new(&a);
                    for k in 0..kmfine {
                        let tt_f = tfine[k] as f64; let mut s_eff = muti[[i, 0]];
                        if a_stateful.eval(tt_f) > muti[[i, 2]] { s_eff += muti[[i, 3]] * (a_stateful.eval(tt_f) - muti[[i, 2]]); }
                        scurve[k] = s_eff;
                    }
                } else { // Non-ecology
                    mcurve = mcurve_ane(cmuti.slice(s![i, 0..3]).as_slice().unwrap(), i, reads.clone(), r.clone(), tfine.clone(), eef.clone(), INITFRAC, KAP, TAUMIN, TAUMAX, t[0]);
                    scurve.fill(muti[[i, 0]]);
                }
                nancf += &mcurve; nsbi += &(&mcurve * &scurve);
            }
        }
        let sbi = &nsbi / &nancf.mapv(|v| v.max(1.0));
        let next_ancf = 1.0 - (&nancf / &nancf.mapv(|v| v.max(1.0)));
        sbmat.row_mut(iter).assign(&sbi); ancmat.row_mut(iter).assign(&next_ancf);
        if iter > 0 { let sbi_prev = sbmat.row(iter-1); let diff = &sbi - &sbi_prev; println!("Convergence check (Sum of Squared Diff in sb): {:.6e}", diff.mapv(|x| x*x).sum()); }
        sb = LinearInterpolator::new(tfine_f64.to_vec(), sbi.to_vec());
        println!("Time for iteration: {:.2?}", start_time.elapsed());
    }
    
    println!("\nAnalysis complete.");

    // --- Write results to CSV files ---
    println!("Writing results to CSV files...");
    write_ndarray_to_csv(&muti, "muti_fits.csv")?;
    println!(" -> muti_fits.csv written successfully.");
    write_ndarray_to_csv(&sbmat, "sbmat_iterations.csv")?;
    println!(" -> sbmat_iterations.csv written successfully.");
    write_ndarray_to_csv(&ancmat, "ancmat_iterations.csv")?;
    println!(" -> ancmat_iterations.csv written successfully.");
    
    // NEW: Write the expanded detailed results file with new headers
    println!("Writing detailed fit results to CSV...");
    let mut writer = csv::Writer::from_path("fit_details.csv")?;
    writer.write_record(&[
        // Neutral model (3 cols)
        "log_prior_n", "p_n_cr0", "x0_n",
        // No-Ecology model (7 cols)
        "log_prior_ne", "p_ne_cr0", "p_ne_cs", "p_ne_ctau", "x0_ne", "s_ne", "tau_ne",
        // Ecology model (11 cols)
        "log_prior_e", "p_e_cr0", "p_e_cs", "p_e_ctau", "p_e_cxc", "p_e_cm", "x0_e", "s_e", "tau_e", "xc_e", "m_e"
    ])?;
    for row in fit_details_mat.rows() {
        writer.write_record(row.iter().map(|&val| val.to_string()))?;
    }
    writer.flush()?;
    println!(" -> fit_details.csv written successfully.");
    
    Ok(())
}