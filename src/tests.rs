// src/tests.rs

// This brings all items from the parent module (main.rs) into scope for testing.
use super::*;
use std::path::Path;
use argmin::core::CostFunction;

// A helper for comparing floating-point numbers.
fn approx_eq(a: f64, b: f64, tolerance: f64) {
    assert!(
        (a - b).abs() < tolerance,
        "assertion failed: `(left ≈ right)`\n  left: `{}`, right: `{}`", a, b
    );
}

/// Helper struct to hold the complex state required for model tests.
struct TestContext {
    reads: Array2<f64>,
    r: Array1<f64>,
    t: Vec<i32>,
    km: usize,
    ee: Array2<f64>,
    eef: LinearInterpolator,
    a: LinearInterpolator,
    ta: LinearInterpolator,
    aif: LinearInterpolator,
}

/// Sets up the full context needed for posterior tests by loading data
/// and recreating the state from iteration 29.
fn setup_test_context_29() -> TestContext {
    // Ensure test data files exist
    assert!(Path::new("test_data/sbmat_iter29.csv").exists(), "Missing test_data/sbmat_iter29.csv");
    assert!(Path::new("test_data/ancmat_iter29.csv").exists(), "Missing test_data/ancmat_iter29.csv");
    
    // Load primary data
    let (reads, r) = data::preprocess_data(PREFIX);

    // Load state from iteration 30
    let sbmat_row = data::load_csv("test_data/sbmat_iter29.csv").unwrap();
    let sbi = sbmat_row.to_shape(sbmat_row.len()).unwrap();
    
    let ancmat_row = data::load_csv("test_data/ancmat_iter29.csv").unwrap();
    let ancf = ancmat_row.to_shape(ancmat_row.len()).unwrap();

    // Recreate time grids and interpolators exactly as in the main loop
    let t: Vec<i32> = (0..=TAUMAX).step_by(G).collect();
    let km = t.len();
    let tfine: Vec<i32> = (TAUMIN..=TAUMAX).collect();
    let tfine_f64: Vec<f64> = tfine.iter().map(|&x| x as f64).collect();

    let sb = LinearInterpolator::new(tfine_f64.clone(), sbi.to_vec());
    let a = LinearInterpolator::new(tfine_f64.clone(), ancf.to_vec());

    let ci: Vec<f64> = (TAUMIN..=TAUMAX).map(|i| {
        let sb_clone = sb.clone();
        let integrand = Arc::new(move |u: f64| sb_clone.eval(u));

        // CORRECTED: Use the exclusive range syntax `..`
        let range = 0.0..i as f64;

        let result = quad_gk!(integrand, range, rel_tol = 1.5e-8);
        result.value
    }).collect();
    let afi: Array1<f64> = Array1::from_vec(ci.iter().map(|&c| (-c).exp()).collect());
    
    let mut ancfs_mono = ancf.clone();
    for i in 1..ancf.len() {
        ancfs_mono[i] = ancfs_mono[i].min(ancfs_mono[i - 1]);
    }

    let mut ta_x: Vec<f64> = Vec::new();
    let mut ta_y: Vec<f64> = Vec::new();
    for i in 0..tfine.len() {
        if ancfs_mono[i] >= 0.9 * XC_MIN && ancfs_mono[i] <= 1.1 * XC_MAX {
            ta_x.push(ancfs_mono[i]);
            ta_y.push(tfine_f64[i]);
        }
    }
    ta_x.reverse();
    ta_y.reverse();
    let ta = LinearInterpolator::new(ta_x, ta_y);
    let tamax = ta.eval(XC_MIN).ceil() as i32;
    
    //let a_integrand = |u: f64| a.eval(u);
    let ai_vec: Vec<f64> = (TAUMIN..=tamax).map(|i| {
        let a_clone = a.clone();
        let a_integrand = Arc::new(move |u: f64| a_clone.eval(u));

        // CORRECTED: Use the exclusive range syntax `..`
        let range = TAUMIN as f64..i as f64;
        
        let result = quad_gk!(a_integrand, range, rel_tol = 1.5e-8);
        result.value
    }).collect();

    let aif = LinearInterpolator::new((TAUMIN..=tamax).map(|i| i as f64).collect(), ai_vec);
    
    let eef = LinearInterpolator::new(tfine_f64.clone(), afi.to_vec());
    
    let v: Array1<f64> = afi.slice(s![(0 - TAUMIN) as usize..]).to_owned();
    let ee = (&v).mapv(|x| 1.0/x).to_shape((v.len(), 1)).unwrap().dot(&v.view().to_shape((1, v.len())).unwrap());

    TestContext { reads, r, t, km, ee, eef, a, ta, aif }
}

/// Sets up the full context needed for posterior tests by loading data
/// and recreating the state from iteration 30.
fn setup_test_context_30() -> TestContext {
    // Ensure test data files exist
    assert!(Path::new("test_data/sbmat_iter30.csv").exists(), "Missing test_data/sbmat_iter30.csv");
    assert!(Path::new("test_data/ancmat_iter30.csv").exists(), "Missing test_data/ancmat_iter30.csv");
    
    // Load primary data
    let (reads, r) = data::preprocess_data(PREFIX);

    // Load state from iteration 30
    let sbmat_row = data::load_csv("test_data/sbmat_iter30.csv").unwrap();
    let sbi = sbmat_row.to_shape(sbmat_row.len()).unwrap();
    
    let ancmat_row = data::load_csv("test_data/ancmat_iter30.csv").unwrap();
    let ancf = ancmat_row.to_shape(ancmat_row.len()).unwrap();

    // Recreate time grids and interpolators exactly as in the main loop
    let t: Vec<i32> = (0..=TAUMAX).step_by(G).collect();
    let km = t.len();
    let tfine: Vec<i32> = (TAUMIN..=TAUMAX).collect();
    let tfine_f64: Vec<f64> = tfine.iter().map(|&x| x as f64).collect();

    let sb = LinearInterpolator::new(tfine_f64.clone(), sbi.to_vec());
    let a = LinearInterpolator::new(tfine_f64.clone(), ancf.to_vec());

    let ci: Vec<f64> = (TAUMIN..=TAUMAX).map(|i| {
        let sb_clone = sb.clone();
        let integrand = Arc::new(move |u: f64| sb_clone.eval(u));

        // CORRECTED: Use the exclusive range syntax `..`
        let range = 0.0..i as f64;

        let result = quad_gk!(integrand, range, rel_tol = 1.5e-8);
        result.value
    }).collect();
    let afi: Array1<f64> = Array1::from_vec(ci.iter().map(|&c| (-c).exp()).collect());
    
    let mut ancfs_mono = ancf.clone();
    for i in 1..ancf.len() {
        ancfs_mono[i] = ancfs_mono[i].min(ancfs_mono[i - 1]);
    }

    let mut ta_x: Vec<f64> = Vec::new();
    let mut ta_y: Vec<f64> = Vec::new();
    for i in 0..tfine.len() {
        if ancfs_mono[i] >= 0.9 * XC_MIN && ancfs_mono[i] <= 1.1 * XC_MAX {
            ta_x.push(ancfs_mono[i]);
            ta_y.push(tfine_f64[i]);
        }
    }
    ta_x.reverse();
    ta_y.reverse();
    let ta = LinearInterpolator::new(ta_x, ta_y);
    let tamax = ta.eval(XC_MIN).ceil() as i32;
    
    //let a_integrand = |u: f64| a.eval(u);
    let ai_vec: Vec<f64> = (TAUMIN..=tamax).map(|i| {
        let a_clone = a.clone();
        let a_integrand = Arc::new(move |u: f64| a_clone.eval(u));

        // CORRECTED: Use the exclusive range syntax `..`
        let range = TAUMIN as f64..i as f64;
        
        let result = quad_gk!(a_integrand, range, rel_tol = 1.5e-8);
        result.value
    }).collect();

    let aif = LinearInterpolator::new((TAUMIN..=tamax).map(|i| i as f64).collect(), ai_vec);
    
    let eef = LinearInterpolator::new(tfine_f64.clone(), afi.to_vec());
    
    let v: Array1<f64> = afi.slice(s![(0 - TAUMIN) as usize..]).to_owned();
    let ee = (&v).mapv(|x| 1.0/x).to_shape((v.len(), 1)).unwrap().dot(&v.view().to_shape((1, v.len())).unwrap());

    TestContext { reads, r, t, km, ee, eef, a, ta, aif }
}


#[test]
fn test_logpg() {
    approx_eq(models::logpg(1200.0, 1500.0, KAP), -12.067638483, 1e-5);
    approx_eq(models::logpg(1500.0, 1200.0, KAP), -11.956066707, 1e-5);
}

#[test]
fn test_conversions() {
    // For convr0, we need the initial read depth R[0]
    let (_, r) = data::preprocess_data(PREFIX);
    let r_t0 = r[0]; // R[1] in Julia
    let r0_ref = 0.0001;
    approx_eq(models::conv_r0(0.1, r_t0, r0_ref, KAP), 0.0001022644, 1e-5);
    approx_eq(models::conv_r0(-0.1, r_t0, r0_ref, KAP), 9.04837418e-5, 1e-5);

    approx_eq(models::conv_s(0.2), 0.141960159, 1e-5);
    approx_eq(models::conv_tau(0.3, TAUMIN, TAUMAX), 55.8264122, 1e-5);
    approx_eq(models::conv_xc(0.4), 0.739475064, 1e-5);
    approx_eq(models::conv_m(0.5), 1.282672729, 1e-5);
}

#[test]
fn test_interpolator_evals() {
    let ctx = setup_test_context_29();

    approx_eq(ctx.eef.eval(0.0), 1.0, 1e-5);
    approx_eq(ctx.eef.eval(10.0), 0.983846894, 1e-5);
    approx_eq(ctx.eef.eval(20.0), 0.777269718, 1e-5);

    approx_eq(ctx.aif.eval(0.0), 19.98462279, 1e-5);
    approx_eq(ctx.aif.eval(10.0), 29.92418442, 1e-5);
    approx_eq(ctx.aif.eval(20.0), 38.90326991, 1e-5);
}

#[test]
fn test_logpg_n() {
    let ctx = setup_test_context_29();
    
    // Test case for lineage 1 (index 0)
    let cost_fn_1 = NeutralCost { i: 0, reads: &ctx.reads, r: &ctx.r, t: &ctx.t, km: ctx.km, ee: &ctx.ee, kap: KAP };
    let logp_1 = -cost_fn_1.cost(&vec![-0.1]).unwrap();
    approx_eq(logp_1, -81.76268902, 1e-3);

    // Test case for lineage 2 (index 1)
    let cost_fn_2 = NeutralCost { i: 1, reads: &ctx.reads, r: &ctx.r, t: &ctx.t, km: ctx.km, ee: &ctx.ee, kap: KAP };
    let logp_2 = -cost_fn_2.cost(&vec![0.0]).unwrap();
    approx_eq(logp_2, -64.40935027, 1e-3);
}

#[test]
fn test_logpg_ane() {
    let ctx = setup_test_context_29();
    let cost_fn = NonEcologyCost {
        i: 1, // lineage 2 (index 1)
        reads: &ctx.reads, r: &ctx.r, t: &ctx.t, km: ctx.km,
        eef: &ctx.eef, kap: KAP, initfrac: INITFRAC, ee: &ctx.ee,
        taumin: TAUMIN, taumax: TAUMAX
    };
    let params = vec![-0.1, 0.3, 0.5];
    let logp = -cost_fn.cost(&params).unwrap();
    approx_eq(logp, -98.31590413, 1e-4);
}

#[test]
fn test_logpg_a() {
    let ctx = setup_test_context_29();
    let cost_fn = EcologyCost {
        i: 6, // lineage 7 (index 6)
        reads: &ctx.reads, r: &ctx.r, t: &ctx.t, km: ctx.km,
        eef: &ctx.eef, a: &ctx.a, ta: &ctx.ta, aif: &ctx.aif,
        kap: KAP, initfrac: INITFRAC, ee: &ctx.ee,
        taumin: TAUMIN, taumax: TAUMAX
    };
    let params = vec![-3e-5, -1.33, -1.35, 0.16, 0.65];
    let logp = -cost_fn.cost(&params).unwrap();
    approx_eq(logp, -104.31346619, 1e-2);

    let ctx = setup_test_context_29();
    let cost_fn = EcologyCost {
        i: 0, // lineage 7 (index 6)
        reads: &ctx.reads, r: &ctx.r, t: &ctx.t, km: ctx.km,
        eef: &ctx.eef, a: &ctx.a, ta: &ctx.ta, aif: &ctx.aif,
        kap: KAP, initfrac: INITFRAC, ee: &ctx.ee,
        taumin: TAUMIN, taumax: TAUMAX
    };
    let params = vec![-0.1, 0.3, -0.5, 0.2, 2.0];
    let logp = -cost_fn.cost(&params).unwrap();
    approx_eq(logp, -24593.64177034579, 0.1);

    let ctx = setup_test_context_29();
    let cost_fn = EcologyCost {
        i: 1, // lineage 7 (index 6)
        reads: &ctx.reads, r: &ctx.r, t: &ctx.t, km: ctx.km,
        eef: &ctx.eef, a: &ctx.a, ta: &ctx.ta, aif: &ctx.aif,
        kap: KAP, initfrac: INITFRAC, ee: &ctx.ee,
        taumin: TAUMIN, taumax: TAUMAX
    };
    let params = vec![-0.1, 0.3, 1.5, 0.2, 0.4];
    let logp = -cost_fn.cost(&params).unwrap();
    approx_eq(logp, -65.5970288539419, 1e-2);
}

#[test]
fn test_mcurve_ane_tail() {
    // 1. Setup the context from iteration 30
    let ctx = setup_test_context_30();
    // Recreate time grids needed by the mcurve function
    let tfine: Vec<i32> = (TAUMIN..=TAUMAX).collect();
    let t0 = ctx.t[0];

    // 2. Define parameters and expected "golden" values for the test
    let params = vec![-0.1, 0.3, 0.5];
    let expected_tail: [f64; 11] = [
        1.4511859068684548e-5, 1.4894536936571837e-5, 1.528242034619445e-5,
        1.5675649708099083e-5, 1.607436138127745e-5, 1.647869258211399e-5,
        1.68887820643942e-5, 1.7304767386695007e-5, 1.7726789836510502e-5,
        1.8154989300051553e-5, 1.8589505643628016e-5,
    ];

    // 3. Call the function for lineage 1 (index 0)
    let result_curve = models::mcurve_ane(
        &params,
        0, // lineage index
        &ctx.reads, &ctx.r, &tfine, &ctx.eef, INITFRAC, KAP, TAUMIN, TAUMAX, t0,
    );

    // 4. Slice the last 11 elements and compare
    let len = result_curve.len();
    assert!(len >= 11, "Result curve is too short");
    let actual_tail = result_curve.slice(s![len - 11..]);

    for (i, (actual, expected)) in actual_tail.iter().zip(expected_tail.iter()).enumerate() {
        println!("ane tail[{}]: actual={}, expected={}", i, actual, expected);
        approx_eq(*actual, *expected, 1e-7); // Using a tolerance that accounts for interpolator differences
    }
}

#[test]
fn test_mcurve_a_tail() {
    // 1. Setup the context from iteration 30
    let ctx = setup_test_context_30();
    let tfine: Vec<i32> = (TAUMIN..=TAUMAX).collect();
    let t0 = ctx.t[0];

    // 2. Define parameters and expected "golden" values for the test
    let params = vec![-3e-5, -1.33, -1.35, 0.16, 0.65];
    let expected_tail: [f64; 11] = [
        0.0003496296440182019, 0.0003287293583102889, 0.00030897967881259156,
        0.00029032846429097645, 0.0002727244565129966, 0.00025611758569188755,
        0.0002404591287751252, 0.0002257017850332482, 0.0002117998334179797,
        0.00019870912268861796, 0.0001863871379159545,
    ];

    // 3. Call the function for lineage 7 (index 6)
    let result_curve = models::mcurve_a(
        &params,
        6, // lineage index
        &ctx.reads, &ctx.r, &tfine, &ctx.eef, &ctx.a, &ctx.ta, &ctx.aif, INITFRAC, KAP,
        TAUMIN, TAUMAX, t0,
    );

    // 4. Slice the last 11 elements and compare
    let len = result_curve.len();
    assert!(len >= 11, "Result curve is too short");
    let actual_tail = result_curve.slice(s![len - 11..]);

    for (i, (actual, expected)) in actual_tail.iter().zip(expected_tail.iter()).enumerate() {
        println!("a tail[{}]: actual={}, expected={}", i, actual, expected);
        // This test will be sensitive to any divergence in the interpolators.
        // We start with a looser tolerance and can tighten it if it passes.
        approx_eq(*actual, *expected, 1e-5); 
    }
}

#[test]
fn debug_ecology_model_components() {
    let ctx = setup_test_context_29();
    
    // --- Test Case Parameters (from logpg_a test) ---
    let params = vec![-3e-5, -1.33, -1.35, 0.16, 0.65];
    let _s = conv_s(params[1]);
    let tau = conv_tau(params[2], TAUMIN, TAUMAX);
    let xc = conv_xc(params[3]);
    let m = conv_m(params[4]);

    let m1 = conv_m(3.5);
    println!("m: {:?}", m1);

    // --- 1. Test the 'ta' interpolator ---
    let ta_eval = ctx.ta.eval(0.7);
    let expected_ta_eval = 23.821615924740012; //  JULIA VALUE
    println!("Rust ta(0.7) = {}", ta_eval);
    approx_eq(ta_eval, expected_ta_eval, 1e-5);

    // --- 2. Test the 'tend' calculation for a specific time step ---
    // Let's use k=9 (t=72), which is the 10th time point.
    let k = 9;
    let tk_f = ctx.t[k] as f64; // t = 72
    
    let a_at_tk = ctx.a.eval(tk_f);
    let expected_a_at_tk = 0.09837513871412995; //  JULIA VALUE
    println!("Rust a(72) = {}", a_at_tk);
    approx_eq(a_at_tk, expected_a_at_tk, 1e-3);

    let tend = if a_at_tk < xc { ctx.ta.eval(xc) } else { tk_f };
    let expected_tend = 22.88961792019225; //  JULIA VALUE
    println!("Rust tend for t=72 is: {}", tend);
    approx_eq(tend, expected_tend, 1e-5);

    // --- 3. Test the final exponent ('integral_term') ---
    let integral_term_val = m * (ctx.aif.eval(tend) - ctx.aif.eval(tau) - xc * (tend - tau));
    let expected_integral_term = 3.6009055809583077; //  JULIA VALUE
    println!("Rust integral term (exponent) for t=72 is: {}", integral_term_val);
    approx_eq(integral_term_val, expected_integral_term, 1e-5);
}