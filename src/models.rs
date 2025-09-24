// src/models.rs
use crate::utils::LinearInterpolator;
use ndarray::{Array1, Array2};
use argmin::core::{CostFunction, Error};
use std::f64::consts::PI;

// Constants for parameter ranges
const S_MIN: f64 = 0.01;
const S_MAX: f64 = 0.25;
const XC_MIN: f64 = 0.5;
const XC_MAX: f64 = 0.9;
const M_MIN: f64 = 0.5;
const M_MAX: f64 = 4.5;

// Parameter conversion functions (from unconstrained to constrained)
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
/// This is the core likelihood component.
pub fn logpg(x: f64, xt: f64, kap: f64) -> f64 {
    let xtp = xt.max(1.0);
    -(xtp + x - 2.0 * (xtp * x).sqrt()) / kap - 0.5 * (4.0 * xtp * kap * PI).ln()
}

// --- Cost Function for Neutral Model ---
#[derive(Clone)]
pub struct NeutralCost<'a> {
    pub i: usize,
    pub reads: &'a Array2<f64>,
    pub r: &'a Array1<f64>,
    pub t: &'a [i32],
    pub km: usize,
    pub ee: &'a Array2<f64>,
    pub kap: f64,
}

impl<'a> CostFunction for NeutralCost<'a> {
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
pub struct NonEcologyCost<'a> {
    pub i: usize, pub reads: &'a Array2<f64>, pub r: &'a Array1<f64>, pub t: &'a [i32], pub km: usize,
    pub eef: &'a LinearInterpolator, pub kap: f64, pub initfrac: f64, pub ee: &'a Array2<f64>,
    pub taumin: i32, pub taumax: i32,
}

impl<'a> CostFunction for NonEcologyCost<'a> {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, p: &Self::Param) -> Result<Self::Output, Error> {
        let x0 = conv_r0(p[0], self.r[0], self.reads[[self.i, 0]] / self.r[0], self.kap);
        let s = conv_s(p[1]);
        let mut tau = conv_tau(p[2], self.taumin, self.taumax);

        if x0.is_nan() {
            return Ok(f64::INFINITY);
        }
        if s.is_nan() {
            return Ok(f64::INFINITY);
        }

        if tau.is_nan() {
            tau = self.taumax as f64;
        }

        let mut logp = 0.0;
        let eef_t1 = self.eef.eval(self.t[0] as f64);
        for k in 0..self.km {
            let tk_f = self.t[k] as f64;
            let neutral_reads = self.r[k] * x0 * self.ee[[0, self.t[k] as usize]];
            let mut mutant_reads = 0.0;
            if tk_f >= tau {
                mutant_reads = self.initfrac * x0 * self.r[k] * self.eef.eval(tk_f) / eef_t1 * (s * (tk_f - tau)).exp();
            }
            logp += logpg(self.reads[[self.i, k]], neutral_reads + mutant_reads, self.kap);
        }

        Ok(-logp)
    }
}

// --- Cost Function for Ecology Mutant Model ---
#[derive(Clone)]
pub struct EcologyCost<'a> {
    pub i: usize, pub reads: &'a Array2<f64>, pub r: &'a Array1<f64>, pub t: &'a [i32], pub km: usize,
    pub eef: &'a LinearInterpolator, pub a: &'a LinearInterpolator, pub ta: &'a LinearInterpolator,
    pub aif: &'a LinearInterpolator, pub kap: f64, pub initfrac: f64, pub ee: &'a Array2<f64>,
    pub taumin: i32, pub taumax: i32,
}

impl<'a> CostFunction for EcologyCost<'a> {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, p: &Self::Param) -> Result<Self::Output, Error> {
        let x0 = conv_r0(p[0], self.r[0], self.reads[[self.i, 0]] / self.r[0], self.kap);
        let s = conv_s(p[1]);
        let mut tau = conv_tau(p[2], self.taumin, self.taumax);
        let xc = conv_xc(p[3]);
        let m = conv_m(p[4]);


        // problem, ask Mike
        if x0.is_nan() {
            return Ok(f64::INFINITY);
        }
        if s.is_nan() {
            return Ok(f64::INFINITY);
        }
        if xc.is_nan() {
            return Ok(f64::INFINITY);
        }
        if m.is_nan() {
            return Ok(f64::INFINITY);
        }
        
        //if tau.is_nan() { return Ok(f64::INFINITY); }
        if tau.is_nan() {
            tau = self.taumax as f64;
        }

        let mut logp = 0.0;
        let eef_t1 = self.eef.eval(self.t[0] as f64);

        for k in 0..self.km {
            let tk_f = self.t[k] as f64;
            let neutral_reads = self.r[k] * x0 * self.ee[[self.t[0] as usize, self.t[k] as usize]];
            let mut mutant_reads = 0.0;
            if tk_f >= tau {
                mutant_reads = self.initfrac * x0 * self.r[k] * self.eef.eval(tk_f) / eef_t1 * (s * (tk_f - tau)).exp();
                if self.a.eval(tau) > xc {
                    let tend = if self.a.eval(tk_f) < xc { self.ta.eval(xc) } else { tk_f };
                    let integral_term = self.aif.eval(tend) - self.aif.eval(tau) - xc * (tend - tau);
                    mutant_reads *= (m * integral_term).exp();
                }
            }

            /*
            if self.i == 6 {
                println!("k: {:?}", k);
                println!("self.reads[[self.i, k]]: {:?}", self.reads[[self.i, k]]);
                println!("mutant: {:?}", mutant_reads);
                println!("neutral: {:?}", neutral_reads);
                println!("logp: {:?}", logpg(self.reads[[self.i, k]], neutral_reads + mutant_reads, self.kap));
            }
            */

            logp += logpg(self.reads[[self.i, k]], neutral_reads + mutant_reads, self.kap);
        }       

        Ok(-logp)
    }
}

// --- Functions to calculate mutant trajectories (mcurve) ---

pub fn mcurve_ane(
    params_c: &[f64], i: usize, reads: &Array2<f64>, r: &Array1<f64>,
    tfine: &[i32], eef: &LinearInterpolator, initfrac: f64, kap: f64,
    taumin: i32, taumax: i32, t0: i32
) -> Array1<f64> {
    let x0 = conv_r0(params_c[0], r[0], reads[[i, 0]] / r[0], kap);
    let s = conv_s(params_c[1]);
    let tau = conv_tau(params_c[2], taumin, taumax);
    let eef_t1 = eef.eval(t0 as f64);

    if x0.is_nan() {
        return Array1::zeros(tfine.len());
    }
    if s.is_nan() {
        return Array1::zeros(tfine.len());
    }

    // issue ask Mike
    if tau.is_nan() { return Array1::zeros(tfine.len()); }

    let mut curve = Array1::zeros(tfine.len());
    for (k, &tk_f_int) in tfine.iter().enumerate() {
        let tk_f = tk_f_int as f64;
        if tk_f >= tau {
            curve[k] = initfrac * x0 * eef.eval(tk_f) / eef_t1 * (s * (tk_f - tau)).exp();
        }
    }
    curve
}

pub fn mcurve_a(
    params_c: &[f64], i: usize, reads: &Array2<f64>, r: &Array1<f64>,
    tfine: &[i32], eef: &LinearInterpolator, a: &LinearInterpolator, ta: &LinearInterpolator,
    aif: &LinearInterpolator, initfrac: f64, kap: f64, taumin: i32, taumax: i32, t0: i32
) -> Array1<f64> {
    let x0 = conv_r0(params_c[0], r[0], reads[[i, 0]] / r[0], kap);
    let s = conv_s(params_c[1]);
    let tau = conv_tau(params_c[2], taumin, taumax);
    let xc = conv_xc(params_c[3]);
    let m = conv_m(params_c[4]);

    if x0.is_nan() {
        return Array1::zeros(tfine.len());
    }
    if s.is_nan() {
        return Array1::zeros(tfine.len());
    }
    if xc.is_nan() {
        return Array1::zeros(tfine.len());
    }
    if m.is_nan() {
        return Array1::zeros(tfine.len());
    }
    // issue ask mike
    if tau.is_nan() { return Array1::zeros(tfine.len()); }

    let eef_t1 = eef.eval(t0 as f64);

    let mut curve = Array1::zeros(tfine.len());
    for (k, &tk_f_int) in tfine.iter().enumerate() {
        let tk_f = tk_f_int as f64;
        if tk_f >= tau {
            let mut rt = initfrac * x0 * eef.eval(tk_f) / eef_t1 * (s * (tk_f - tau)).exp();
            if a.eval(tau) > xc {
                let tend = if a.eval(tk_f) < xc { ta.eval(xc) } else { tk_f };
                let integral_term = aif.eval(tend) - aif.eval(tau) - xc * (tend - tau);
                rt *= (m * integral_term).exp();
            }
            curve[k] = rt;
        }
    }
    curve
}