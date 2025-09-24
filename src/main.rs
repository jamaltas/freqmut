// src/main.rs
mod data;
mod utils;
mod models;
mod neldermead;

use ndarray::{arr2, s, Array1, Array2, ArrayView1, Axis};
use rayon::prelude::*;
use std::time::Instant;
use argmin::core::{CostFunction};
//use argmin::solver::neldermead::NelderMead;
use crate::utils::{LinearInterpolator, median};
use crate::models::*;
use crate::neldermead::*;
use std::error::Error;

use quad_gk::quad_gk;
use quad_gk::GKConfig;
use std::sync::Arc;

use egobox_doe::{Lhs, LhsKind, SamplingMethod};

// --- Global constants ---
const PREFIX: &str = "eco15";
//const PREFIX: &str = "levy_exp";
const TAUMIN: i32 = -20;
const TAUMAX: i32 = 112;
const G: usize = 8;
const INITFRAC: f64 = 0.05;
const NEPENALTY: f64 = 5.0;
const APENALTY: f64 = 20.0;
const KAP: f64 = 2.5;
const NITER: usize = 2;

// Parameter ranges
const XC_MIN: f64 = 0.5;
const XC_MAX: f64 = 0.9;

/// Writes a 2D ndarray of f64 to a CSV file.
fn write_ndarray_to_csv(data: &Array2<f64>, path: &str) -> Result<(), Box<dyn Error>> {
    let mut writer = csv::Writer::from_path(path)?;
    for row in data.rows() {
        writer.write_record(row.iter().map(|&val| val.to_string()))?;
    }
    writer.flush()?;
    Ok(())
}

/// Creates an initial simplex as a fixed-size array for the Nelder-Mead algorithm.
/// This version is generic over the dimensions to directly produce the required type.
fn make_simplex<const DIMS: usize, const POINTS: usize>(
    initial_point: &[f64; DIMS]
) -> [[f64; DIMS]; POINTS] {
    // Compile-time check would be ideal, but a runtime assert is clear and effective.
    assert_eq!(POINTS, DIMS + 1, "The number of simplex points must be DIMS + 1.");

    // Parameters from Optim.AffineSimplexer(a, b)
    let a: f64 = 0.00025;
    let b: f64 = 0.05;

    // Create the result array, initialized with zeros. `MaybeUninit` could also be used
    // for a slight performance gain if DIMS is very large, but this is simpler.
    let mut simplex = [[0.0; DIMS]; POINTS];

    // The first vertex is the initial point itself.
    simplex[0] = *initial_point;

    // Generate the other DIMS vertices
    for i in 0..DIMS {
        let mut next_point = *initial_point;
        
        // Apply the affine transformation formula: new_xi = original_xi * (1 + b) + a
        let new_xi = initial_point[i] * (1.0 + b) + a;

        // Update the i-th coordinate of the new vertex.
        next_point[i] = new_xi;
        
        // Place the new point in the simplex. Note the `i + 1` index.
        simplex[i + 1] = next_point;
    }
    
    simplex
}


fn main() {
    // 1. Load and preprocess data
    println!("Loading and preprocessing data...");
    let (reads, r) = data::preprocess_data(PREFIX);
    let nlin = reads.nrows();

    // 2. Setup time grids and initial parameters
    let t: Vec<i32> = (0..=TAUMAX).step_by(G).collect();
    let km = t.len();
    let tfine: Vec<i32> = (TAUMIN..=TAUMAX).collect();
    let kmfine = tfine.len();
    let tfine_f64: Vec<f64> = tfine.iter().map(|&x| x as f64).collect();

    // Define parameter bounds for the 3-parameter Non-Ecology model
    let xlimits_ne = arr2(&[
        [-1.5, 0.5],  // Range for x0
        [-1.5, 1.5],  // Range for s
        [-1.5, 1.0],  // Range for tau
    ]);

    let initgridne = Lhs::new(&xlimits_ne).kind(LhsKind::Centered).sample(250);
    
    /*
    // Initial guesses for optimization
    let initgridne = Array2::from_shape_vec((5, 3), vec![
        -0.01, -0.2, -3.5,
        -0.01, -0.2, -2.5,
        -0.01, -0.2, -1.5,
        -0.01, -0.2, -0.5,
        -0.01, -0.2,  0.5,
    ]).unwrap();
    */


    // Define parameter bounds for the 5-parameter Ecology model
    let xlimits_a = arr2(&[
        [-1.5, 0.5],  // Range for x0
        [-1.5, 1.5],  // Range for s
        [-1.5, 1.0],  // Range for tau
        [-1.5, 1.5],  // Range for xc
        [-1.5, 1.5],  // Range for m
    ]);

    let initgrida = Lhs::new(&xlimits_a).kind(LhsKind::Centered).sample(250);

    /*
    let initgrida = Array2::from_shape_vec((15, 5), vec![
        -0.01, -0.2, -3.5, 0.01, -1.0,
        -0.01, -0.2, -2.5, 0.01, -1.0,
        -0.01, -0.2, -1.5, 0.01, -1.0,
        -0.01, -0.2, -0.5, 0.01, -1.0,
        -0.01, -0.2, 0.5, 0.01, -1.0,
        -0.01, -0.2, -3.5, 3.5, -1.0,
        -0.01, -0.2, -2.5, 3.5, -1.0,
        -0.01, -0.2, -1.5, 3.5, -1.0,
        -0.01, -0.2, -0.5, 3.5, -1.0,
        -0.01, -0.2, 0.5, 3.5, -1.0,
        -0.01, -0.2, -3.5, -3.5, -1.0,
        -0.01, -0.2, -2.5, -3.5, -1.0,
        -0.01, -0.2, -1.5, -3.5, -1.0,
        -0.01, -0.2, -0.5, -3.5, -1.0,
        -0.01, -0.2, 0.5, -3.5, -1.0,
    ]).unwrap();
    */

    // 3. Initial guess for mean fitness (sb)
    let first_col_mean = reads.column(0).mean().unwrap();

    let putneut_indices: Vec<usize> = (0..nlin)
        .filter(|&i| {
            let first = reads[[i, 0]];
            let last  = reads[[i, km - 1]];
            first < first_col_mean && first > last && last > 1.0
        })
        .collect();

    let mut sbic = Array1::zeros(km);

    for k in 1..km {
        // Collect normalized column values for selected rows:
        let mut col_k_vals: Vec<f64> = putneut_indices.iter()
            .map(|&i| reads[[i, k]] / r[k]) // r[k] corresponds to R[k]
            .collect();
        let mut col_km1_vals: Vec<f64> = putneut_indices.iter()
            .map(|&i| reads[[i, k - 1]] / r[k - 1]) // previous column normalization
            .collect();

        // Use your median() function (which takes &mut Vec<f64>):
        let median_k = median(&mut col_k_vals);
        let median_km1 = median(&mut col_km1_vals);

        let ratio = median_k / median_km1;
        let dt = (t[k] as f64) - (t[k - 1] as f64);
        let s_val = -ratio.ln() / dt;

        sbic[k] = s_val.max(sbic[k - 1]); // same as Julia’s max(...)
    }

    // Build x points:
    let mut sb_t_pts: Vec<f64> = Vec::with_capacity(1 + t.len());
    sb_t_pts.push(TAUMIN as f64);
    sb_t_pts.extend(t.iter().map(|&x| x as f64));

    // Build y points: 0.0, then sbic (like Julia’s vcat):
    let mut sb_y_pts: Vec<f64> = Vec::with_capacity(1 + sbic.len());
    sb_y_pts.push(0.0);
    sb_y_pts.extend(sbic.iter().cloned());

    let mut sb = LinearInterpolator::new(sb_t_pts, sb_y_pts);
    
    // 4. Main iteration loop
    let mut sbmat = Array2::zeros((NITER, kmfine));
    let mut ancmat = Array2::zeros((NITER, kmfine));
    let mut muti = Array2::zeros((nlin, 4));

    //let mut sbi = Array1::from_vec(tfine.iter().map(|&time| sb.eval(time as f64)).collect());
    //let mut a = empty_interpolator();

    for iter in 0..NITER {
        let start_time = Instant::now();
        println!("Iteration: {}", iter + 1);

        // --- Update population-level variables for this iteration ---
        let ci: Vec<f64> = (TAUMIN..=TAUMAX).map(|i| {
            // The quad_gk macro requires a cloneable closure, so we clone the interpolator.
            let sb_clone = sb.clone();
            let integrand = Arc::new(move |u: f64| sb_clone.eval(u));

            // Define the integration range. quad_gk handles inverted ranges automatically.
            let range = 0.0..i as f64;

            // Call the macro with a relative tolerance matching Julia's default.
            let result = quad_gk!(integrand, range, rel_tol = 1.5e-8);
            result.value
        }).collect();

        let afi: Array1<f64> = Array1::from_vec(ci.iter().map(|&c| (-c).exp()).collect());
        
        let ancf = if iter == 0 { afi.clone() } else { ancmat.row(iter - 1).to_owned() };
        
        let a = LinearInterpolator::new(tfine_f64.clone(), ancf.to_vec());

        let mut ancfs_mono = ancf.clone();
        for i in 1..ancf.len() {
            ancfs_mono[i] = ancfs_mono[i].min(ancfs_mono[i - 1]);
        }
        
        let mut ta_x: Vec<f64> = Vec::new();
        let mut ta_y: Vec<f64> = Vec::new();
        for i in 0..kmfine {
            if ancfs_mono[i] >= 0.9 * XC_MIN && ancfs_mono[i] <= 1.1 * XC_MAX {
                ta_x.push(ancfs_mono[i]);
                ta_y.push(tfine_f64[i]);
            }
        }
        ta_x.reverse();
        ta_y.reverse();
        let ta = LinearInterpolator::new(ta_x, ta_y);
        let tamax = ta.eval(XC_MIN).ceil() as i32;
        
        let ai_vec: Vec<f64> = (TAUMIN..=tamax).map(|i| {
            let a_clone = a.clone();
            let a_integrand = Arc::new(move |u: f64| a_clone.eval(u));

            let range = TAUMIN as f64..i as f64;
            
            let result = quad_gk!(a_integrand, range, rel_tol = 1.5e-8);
            result.value
        }).collect();
        let aif = LinearInterpolator::new((TAUMIN..=tamax).map(|i| i as f64).collect(), ai_vec);
        
        let eef = LinearInterpolator::new(tfine_f64.clone(), afi.to_vec());

        let v: Array1<f64> = afi.slice(s![(0 - TAUMIN) as usize..]).to_owned();
        let recip = v.mapv(|x| 1.0 / x);
        let ee = recip.insert_axis(Axis(1)) * v.insert_axis(Axis(0));

        // --- Parallel fitting for each lineage with improved sequential logic ---
        let results: Vec<(Array1<f64>, Array1<f64>)> = (0..nlin)
        //.into_par_iter()
        .into_iter()
        .map(|i| {
            // Re-bind variables for closure
            let (eef, a, ta, aif, ee) = (eef.clone(), a.clone(), ta.clone(), aif.clone(), ee.clone());

            // --- 1. Fit Neutral Model ---
            let cost_n = |p: &[f64; 1]| NeutralCost { i, reads: &reads, r: &r, t: &t, km, ee: &ee, kap: KAP}.cost(&p.to_vec());
            let simplex_n = make_simplex(&[-2.0]);
            let mut solver_n = NelderMead::<1, 2>::new(simplex_n)
                    .with_alpha(1.0).unwrap()
                    .with_gamma(2.5).unwrap()
                    .with_rho(0.4).unwrap()
                    .with_sigma(0.4).unwrap();
            let res = solver_n.run(&cost_n, 500, 1e-6);
            //let res_n = Executor::new(cost_n, solver_n)
                //.configure(|s| s.max_iters(500))
                //.run()
                //.expect("Neutral optimization failed");
            let min_n = res.best_cost;
            
            // --- 2. Fit Non-Ecology Model ---
            let cost_ne_fn = NonEcologyCost { i, reads: &reads, r: &r, t: &t, km, eef: &eef, kap: KAP, initfrac: INITFRAC, ee: &ee, taumin: TAUMIN, taumax: TAUMAX };
            let mut best_res_ne: Option<OptimizationResult<3>> = None;
            for guess in initgridne.rows() {

                let mut guess_array = [0.0; 3];
                for (i, val) in guess.iter().enumerate() {
                    guess_array[i] = *val;
                }

                //let guess_array: [f64; 3] = guess.to_slice().unwrap().try_into().expect("Should be length 3");
                let cost_ne = |p: &[f64; 3]| cost_ne_fn.cost(&p.to_vec()); 

                let simplex: [[f64; 3]; _] = make_simplex(&guess_array);
                let mut solver = NelderMead::<3, 4>::new(simplex)
                    .with_alpha(1.0).unwrap()
                    .with_gamma(2.5).unwrap()
                    .with_rho(0.4).unwrap()
                    .with_sigma(0.4).unwrap();
                let res = solver.run(&cost_ne, 2500, 1e-6);
                //let res = Executor::new(cost_ne_fn.clone(), solver)
                    //.configure(|s| s.max_iters(2500))
                    //.run().unwrap();
                if best_res_ne.is_none() || res.best_cost < best_res_ne.as_ref().unwrap().best_cost {
                    best_res_ne = Some(res);
                }
            }
            let best_res_ne = best_res_ne.unwrap();
            let min_ne = best_res_ne.best_cost;

            // --- 3. Compare Neutral vs Non-Ecology and decide whether to proceed ---
            if min_n <= min_ne + NEPENALTY {
                // Neutral model is better or not significantly worse, so we stop here.
                return (Array1::zeros(4), Array1::zeros(5));
            }

            // --- 4. If Non-Ecology is better, fit Ecology Model ---
            let cost_a_fn = EcologyCost { i, reads: &reads, r: &r, t: &t, km, eef: &eef, a: &a, ta: &ta, aif: &aif, kap: KAP, initfrac: INITFRAC, ee: &ee, taumin: TAUMIN, taumax: TAUMAX };
            let mut best_res_a: Option<OptimizationResult<5>> = None;
            for guess in initgrida.rows() {

                let mut guess_array = [0.0; 5];
                for (i, val) in guess.iter().enumerate() {
                    guess_array[i] = *val;
                }
                
                //let guess_array: [f64; 5] = guess.to_slice().unwrap().try_into().expect("Should be length 5");
                let cost_a = |p: &[f64; 5]| cost_a_fn.cost(&p.to_vec()); 
                
                let simplex: [[f64; 5]; _] = make_simplex(&guess_array);
                let mut solver = NelderMead::<5,6>::new(simplex)
                    .with_alpha(1.0).unwrap()
                    .with_gamma(2.5).unwrap()
                    .with_rho(0.4).unwrap()
                    .with_sigma(0.4).unwrap();
                let res = solver.run(&cost_a, 2500, 1e-6);
                
                //let res = Executor::new(cost_a_fn.clone(), solver)
                    //.configure(|s| s.max_iters(2500))
                    //.run().unwrap();
                
                /*
                if i == 6 {
                    // We use format! to left-align the guess for cleaner output
                    println!(
                        "  -> Initial guess {:.10} resulted in cost: {:.2}",
                        format!("{:?}", guess.to_vec().as_slice()),
                        res.best_cost
                    );
                }
                */

                if best_res_a.is_none() || res.best_cost < best_res_a.as_ref().unwrap().best_cost {
                    best_res_a = Some(res);
                }
            }
            let best_res_a = best_res_a.unwrap();
            let min_a = best_res_a.best_cost;

            // --- 5. Compare Non-Ecology vs Ecology and finalize parameters ---
            let mut muti_row = Array1::zeros(4);
            let mut cmuti_row = Array1::zeros(5);

            if min_a + APENALTY < min_ne {
                // Ecology model is significantly better
                let p = best_res_a.best_param;
                muti_row[0] = conv_s(p[1]);
                muti_row[1] = conv_tau(p[2], TAUMIN, TAUMAX);
                muti_row[2] = conv_xc(p[3]);
                muti_row[3] = conv_m(p[4]);
                let p_view = ArrayView1::from(&p);
                cmuti_row.assign(&p_view);
            } else {
                // Non-Ecology model remains the winner
                let p = best_res_ne.best_param;
                muti_row[0] = conv_s(p[1]);
                muti_row[1] = conv_tau(p[2], TAUMIN, TAUMAX);
                //let p_view = ArrayView1::from(&p);
                cmuti_row.slice_mut(s![0..3]).assign(&ArrayView1::from(&p));
            }         

            (muti_row, cmuti_row)

        }).collect();

        // --- Update sbi and ancf for next iteration ---
        let mut cmuti = Array2::zeros((nlin, 5));
        for (i, (muti_r, cmuti_r)) in results.into_iter().enumerate() {
            muti.row_mut(i).assign(&muti_r);
            cmuti.row_mut(i).assign(&cmuti_r);
        }

        let mut nsbi = Array1::zeros(kmfine);
        let mut nancf = Array1::zeros(kmfine);

        for i in 0..nlin {
            if muti.row(i).iter().any(|&x| x != 0.0) {
                let mcurve;
                let mut scurve = Array1::zeros(kmfine);
                if muti[[i, 2]] != 0.0 || muti[[i, 3]] != 0.0 { // Ecology mutant
                    mcurve = mcurve_a(cmuti.row(i).as_slice().unwrap(), i, &reads, &r, &tfine, &eef, &a, &ta, &aif, INITFRAC, KAP, TAUMIN, TAUMAX, t[0]);
                    for (k, &tt) in tfine.iter().enumerate() {
                        let tt_f = tt as f64;
                        let mut s_eff = muti[[i, 0]];
                        if a.eval(tt_f) > muti[[i, 2]] {
                            s_eff += muti[[i, 3]] * (a.eval(tt_f) - muti[[i, 2]]);
                        }
                        scurve[k] = s_eff;
                    }
                } else { // Non-ecology mutant
                    mcurve = mcurve_ane(cmuti.slice(s![i, 0..3]).as_slice().unwrap(), i, &reads, &r, &tfine, &eef, INITFRAC, KAP, TAUMIN, TAUMAX, t[0]);
                    scurve.fill(muti[[i, 0]]);
                }
                nancf += &mcurve;
                nsbi += &(&mcurve * &scurve);
            }
        }
        
        let sbi = &nsbi / &nancf.mapv(|v| v.max(1.0));
        let next_ancf = 1.0 - (&nancf / &nancf.mapv(|v| v.max(1.0)));

        sbmat.row_mut(iter).assign(&sbi);
        ancmat.row_mut(iter).assign(&next_ancf);

        // ===================================================================
        // =================== NEW: CONVERGENCE CHECK BLOCK ==================
        // ===================================================================
        if iter > 0 {
            // Get the sb vector from the previous iteration
            let sbi_previous = sbmat.row(iter - 1);
            
            // Calculate the sum of squared differences
            let diff = &sbi - &sbi_previous;
            let sum_sq_diff = diff.mapv(|x| x * x).sum();
            
            // Print the result, using scientific notation for clarity
            println!("Convergence check (Sum of Squared Diff in sb): {:.6e}", sum_sq_diff);
        }
        // ===================================================================
        // ======================= END OF NEW BLOCK ==========================
        // ===================================================================
        
        sb = LinearInterpolator::new(tfine_f64.clone(), sbi.to_vec());

        println!("Time for iteration: {:.2?}", start_time.elapsed());
    }
    
    println!("\nAnalysis complete.");

    // --- Write results to CSV files ---
    println!("Writing results to CSV files...");

    write_ndarray_to_csv(&muti, "muti_fits.csv")
        .expect("Failed to write muti_fits.csv");
    println!(" -> muti_fits.csv written successfully.");

    write_ndarray_to_csv(&sbmat, "sbmat_iterations.csv")
        .expect("Failed to write sbmat_iterations.csv");
    println!(" -> sbmat_iterations.csv written successfully.");

    write_ndarray_to_csv(&ancmat, "ancmat_iterations.csv")
        .expect("Failed to write ancmat_iterations.csv");
    println!(" -> ancmat_iterations.csv written successfully.");
}

#[cfg(test)]
mod tests;