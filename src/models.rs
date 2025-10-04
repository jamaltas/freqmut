// src/models.rs
use crate::utils::{LinearInterpolator, StatefulInterpolator}; // Import StatefulInterpolator
use ndarray::{Array1, Array2};
use argmin::core::{CostFunction, Error};
use std::f64::consts::PI;
use std::sync::Arc; // Import Arc

// Constants for parameter ranges (no changes)
const S_MIN: f64 = 0.01;
const S_MAX: f64 = 0.25;
const XC_MIN: f64 = 0.3;
const XC_MAX: f64 = 0.9;
const M_MIN: f64 = 0.3;
const M_MAX: f64 = 4.5;

// Parameter conversion functions (no changes)
pub fn conv_r0(cr0: f64, r_t0: f64, r0_ref: f64, kap: f64) -> f64 {
    if cr0 > 0.0 {
        r0_ref + (1.0 - (-cr0).exp()) * 3.0 * (0.5 * kap * (4.0 * r0_ref * r_t0 + 3.0 * kap)).sqrt() / r_t0
    } else {
        r0_ref * cr0.exp()
    }
}
pub fn conv_s(cs: f64) -> f64 { (cs.exp() / (1.0 + cs.exp())) * (S_MAX - S_MIN) + S_MIN }
pub fn conv_tau(ctau: f64, taumin: i32, taumax: i32) -> f64 { (ctau.exp() / (1.0 + ctau.exp())) * (taumax - taumin) as f64 + taumin as f64 }
pub fn conv_xc(cxc: f64) -> f64 { (cxc.exp() / (1.0 + cxc.exp())) * (XC_MAX - XC_MIN) + XC_MIN }
pub fn conv_m(cm: f64) -> f64 { (cm.exp() / (1.0 + cm.exp())) * (M_MAX - M_MIN) + M_MIN }

/// Log of Poisson-Gamma distribution probability mass function.
/// This is the core likelihood component. (no changes)
pub fn logpg(x: f64, xt: f64, kap: f64) -> f64 {
    let xtp = xt.max(1.0);
    -(xtp + x - 2.0 * (xtp * x).sqrt()) / kap - 0.5 * (4.0 * xtp * kap * PI).ln()
}

// --- Cost Function for Neutral Model ---
#[derive(Clone)]
pub struct NeutralCost { // Removed 'a lifetime due to Arc
    pub i: usize,
    pub reads: Arc<Array2<f64>>, // Changed to Arc
    pub r: Arc<Array1<f64>>,     // Changed to Arc
    pub t: Arc<Vec<i32>>,        // Changed to Arc
    pub km: usize,
    pub ee: Arc<Array2<f64>>,    // Changed to Arc
    pub kap: f64,
}

impl CostFunction for NeutralCost { // Removed 'a lifetime
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, p: &Self::Param) -> Result<Self::Output, Error> {
        let x0 = conv_r0(p[0], self.r[0], self.reads[[self.i, 0]] / self.r[0], self.kap);
        let mut logp = 0.0;
        for k in 0..self.km {
            let tk = self.t[k] as usize;
            let expected_reads = self.r[k] * x0 * self.ee[[0, tk]];
            logp += logpg(self.reads[[self.i, k]], expected_reads, self.kap);
        }
        Ok(-logp)
    }
}

// --- Cost Function for Non-Ecology Mutant Model ---
#[derive(Clone)]
pub struct NonEcologyCost { // Removed 'a lifetime
    pub i: usize, pub reads: Arc<Array2<f64>>, pub r: Arc<Array1<f64>>, pub t: Arc<Vec<i32>>, pub km: usize,
    pub eef: Arc<LinearInterpolator>, // Changed to Arc
    pub kap: f64, pub initfrac: f64, pub ee: Arc<Array2<f64>>, // Changed to Arc
    pub taumin: i32, pub taumax: i32,
}

impl CostFunction for NonEcologyCost { // Removed 'a lifetime
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, p: &Self::Param) -> Result<Self::Output, Error> {
        let x0 = conv_r0(p[0], self.r[0], self.reads[[self.i, 0]] / self.r[0], self.kap);
        let s = conv_s(p[1]);
        let mut tau = conv_tau(p[2], self.taumin, self.taumax);

        if x0.is_nan() || s.is_nan() {
            return Ok(f64::INFINITY);
        }
        if tau.is_nan() {
            tau = self.taumax as f64;
        }

        let mut logp = 0.0;
        // Use StatefulInterpolator for 'eef' as tk_f is sequential
        let mut eef_stateful = StatefulInterpolator::new(&self.eef); // Derefs Arc to &LinearInterpolator
        let eef_t1 = eef_stateful.eval(self.t[0] as f64); // eval updates internal hint

        for k in 0..self.km {
            let tk_f = self.t[k] as f64;
            let neutral_reads = self.r[k] * x0 * self.ee[[0, self.t[k] as usize]];
            let mut mutant_reads = 0.0;
            if tk_f >= tau {
                mutant_reads = self.initfrac * x0 * self.r[k] * eef_stateful.eval(tk_f) / eef_t1 * (s * (tk_f - tau)).exp();
            }
            logp += logpg(self.reads[[self.i, k]], neutral_reads + mutant_reads, self.kap);
        }

        Ok(-logp)
    }
}

// --- Cost Function for Ecology Mutant Model ---
#[derive(Clone)]
pub struct EcologyCost { // Removed 'a lifetime
    pub i: usize, pub reads: Arc<Array2<f64>>, pub r: Arc<Array1<f64>>, pub t: Arc<Vec<i32>>, pub km: usize,
    pub eef: Arc<LinearInterpolator>, pub a: Arc<LinearInterpolator>, pub ta: Arc<LinearInterpolator>,
    pub aif: Arc<LinearInterpolator>, pub kap: f64, pub initfrac: f64, pub ee: Arc<Array2<f64>>,
    pub taumin: i32, pub taumax: i32,
}

impl CostFunction for EcologyCost { // Removed 'a lifetime
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, p: &Self::Param) -> Result<Self::Output, Error> {
        let x0 = conv_r0(p[0], self.r[0], self.reads[[self.i, 0]] / self.r[0], self.kap);
        let s = conv_s(p[1]);
        let mut tau = conv_tau(p[2], self.taumin, self.taumax);
        let xc = conv_xc(p[3]);
        let m = conv_m(p[4]);

        if x0.is_nan() || s.is_nan() || xc.is_nan() || m.is_nan() {
            return Ok(f64::INFINITY);
        }
        if tau.is_nan() {
            tau = self.taumax as f64;
        }

        let mut logp = 0.0;
        
        // Use StatefulInterpolator for 'eef' and 'a' as tk_f is sequential.
        // 'ta' and 'aif' might also benefit from StatefulInterpolator for 'tend' and 'tau'.
        let mut eef_stateful = StatefulInterpolator::new(&self.eef);
        let mut a_stateful = StatefulInterpolator::new(&self.a);
        let mut aif_stateful = StatefulInterpolator::new(&self.aif);

        let eef_t1 = eef_stateful.eval(self.t[0] as f64); // Prime the hint

        for k in 0..self.km {
            let tk_f = self.t[k] as f64;
            let neutral_reads = self.r[k] * x0 * self.ee[[self.t[0] as usize, self.t[k] as usize]];
            let mut mutant_reads = 0.0;
            if tk_f >= tau {
                mutant_reads = self.initfrac * x0 * self.r[k] * eef_stateful.eval(tk_f) / eef_t1 * (s * (tk_f - tau)).exp();
                
                // For a.eval(tau), tau is constant for this `cost` call.
                // Call directly on the Arc's LinearInterpolator, which uses O(1) if uniform.
                if self.a.eval(tau) > xc {
                    // a_at_tk_f uses stateful for sequential tk_f
                    let a_at_tk_f = a_stateful.eval(tk_f);
                    
                    // tend depends on a_at_tk_f and ta.eval(xc)
                    // ta.eval(xc): xc is constant for this `cost` call. ta's x_pts are NOT uniform.
                    // So, calling ta_stateful.eval(xc) might reset the hint for each call.
                    // It's probably better to call directly on the underlying Arc for this non-sequential access.
                    let ta_at_xc = self.ta.eval(xc); 
                    let tend = if a_at_tk_f < xc { ta_at_xc } else { tk_f };
                    
                    // aif.eval(tend): tend can be tk_f (sequential) or ta_at_xc (non-sequential relative to tk_f).
                    // If tend is often tk_f, stateful is good. If tend is often ta_at_xc, it might jump.
                    // For safety, let's use stateful for `tend` (as it *can* be sequential)
                    // and direct call for `tau` (as it's definitely non-sequential).
                    let integral_term = aif_stateful.eval(tend) - self.aif.eval(tau) - xc * (tend - tau);
                    mutant_reads *= (m * integral_term).exp();
                }
            }

            logp += logpg(self.reads[[self.i, k]], neutral_reads + mutant_reads, self.kap);
        }       

        Ok(-logp)
    }
}

// --- Functions to calculate mutant trajectories (mcurve) ---

pub fn mcurve_ane(
    params_c: &[f64], i: usize, reads: Arc<Array2<f64>>, r: Arc<Array1<f64>>, // Arcs
    tfine: Arc<Vec<i32>>, eef: Arc<LinearInterpolator>, initfrac: f64, kap: f64, // Arcs
    taumin: i32, taumax: i32, t0: i32
) -> Array1<f64> {
    let x0 = conv_r0(params_c[0], r[0], reads[[i, 0]] / r[0], kap);
    let s = conv_s(params_c[1]);
    let tau = conv_tau(params_c[2], taumin, taumax);
    let t_len = tfine.len();

    if x0.is_nan() || s.is_nan() || tau.is_nan() { // Simplified nan checks
        return Array1::zeros(t_len);
    }

    let mut eef_stateful = StatefulInterpolator::new(&eef);
    let eef_t1 = eef_stateful.eval(t0 as f64);

    let mut curve = Array1::zeros(t_len);
    for k in 0..t_len { // Loop over indices of tfine
        let tk_f = tfine[k] as f64; // Access tfine directly
        if tk_f >= tau {
            curve[k] = initfrac * x0 * eef_stateful.eval(tk_f) / eef_t1 * (s * (tk_f - tau)).exp();
        }
    }
    curve
}

pub fn mcurve_a(
    params_c: &[f64], i: usize, reads: Arc<Array2<f64>>, r: Arc<Array1<f64>>, // Arcs
    tfine: Arc<Vec<i32>>, eef: Arc<LinearInterpolator>, a: Arc<LinearInterpolator>, // Arcs
    ta: Arc<LinearInterpolator>, aif: Arc<LinearInterpolator>, initfrac: f64, kap: f64, // Arcs
    taumin: i32, taumax: i32, t0: i32
) -> Array1<f64> {
    let x0 = conv_r0(params_c[0], r[0], reads[[i, 0]] / r[0], kap);
    let s = conv_s(params_c[1]);
    let tau = conv_tau(params_c[2], taumin, taumax);
    let xc = conv_xc(params_c[3]);
    let m = conv_m(params_c[4]);
    let t_len = tfine.len();


    if x0.is_nan() || s.is_nan() || xc.is_nan() || m.is_nan() || tau.is_nan() { // Simplified nan checks
        return Array1::zeros(t_len);
    }

    let mut eef_stateful = StatefulInterpolator::new(&eef);
    let mut a_stateful = StatefulInterpolator::new(&a);
    let ta_at_xc_val = ta.eval(xc); // xc is constant, call direct
    let mut aif_stateful = StatefulInterpolator::new(&aif);

    let eef_t1 = eef_stateful.eval(t0 as f64);

    let mut curve = Array1::zeros(t_len);
    for k in 0..t_len { // Loop over indices of tfine
        let tk_f = tfine[k] as f64; // Access tfine directly
        if tk_f >= tau {
            let mut rt = initfrac * x0 * eef_stateful.eval(tk_f) / eef_t1 * (s * (tk_f - tau)).exp();
            
            if a.eval(tau) > xc { // tau is constant, call direct
                let a_at_tk_f = a_stateful.eval(tk_f);
                let tend = if a_at_tk_f < xc { ta_at_xc_val } else { tk_f };
                
                let integral_term = aif_stateful.eval(tend) - aif.eval(tau) - xc * (tend - tau);
                rt *= (m * integral_term).exp();
            }
            curve[k] = rt;
        }
    }
    curve
}