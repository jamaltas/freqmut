// src/models/parameters.rs

//! Defines model parameter bounds and conversion functions.

#[derive(Clone, Copy)]
pub struct ModelParameterBounds {
    pub s_min: f64,
    pub s_max: f64,
    pub tau_min: i32,
    pub tau_max: i32,
    pub xc_min: f64,
    pub xc_max: f64,
    pub m_min: f64,
    pub m_max: f64,
    pub r0_rel_min: f64,
    pub r0_rel_max: f64,
}

/// Converts an unconstrained value (e.g., from an optimizer) to a constrained value
/// using a logistic function.
fn logistic_transform(val: f64, min: f64, max: f64) -> f64 {
    (val.exp() / (1.0 + val.exp())) * (max - min) + min
}

/*
pub fn conv_r0(cr0: f64, r_ref: f64, bounds: &ModelParameterBounds) -> f64 {
    let r0_min = r_ref * bounds.r0_rel_min;
    let r0_max = r_ref * bounds.r0_rel_max;
    logistic_transform(cr0, r0_min, r0_max)
}
*/

pub fn conv_r0(cr0: f64, r_ref: f64, kap: f64, r1: f64) -> f64 {
    if cr0 > 0.0 {
        // This is the positive `cr0` branch from the Julia code.
        let adjustment_term = (1.0 - (-cr0).exp()) * 3.0
            * (0.5 * kap * (4.0 * r_ref * r1 + 3.0 * kap)).sqrt()
            / r1;
        r_ref + adjustment_term
    } else {
        // This is the non-positive `cr0` branch.
        r_ref * cr0.exp()
    }
}

pub fn conv_s(cs: f64, bounds: &ModelParameterBounds) -> f64 {
    logistic_transform(cs, bounds.s_min, bounds.s_max)
}

pub fn conv_tau(ctau: f64, bounds: &ModelParameterBounds) -> f64 {
    logistic_transform(ctau, bounds.tau_min as f64, bounds.tau_max as f64)
}

pub fn conv_xc(cxc: f64, bounds: &ModelParameterBounds) -> f64 {
    logistic_transform(cxc, bounds.xc_min, bounds.xc_max)
}

pub fn conv_m(cm: f64, bounds: &ModelParameterBounds) -> f64 {
    logistic_transform(cm, bounds.m_min, bounds.m_max)
}