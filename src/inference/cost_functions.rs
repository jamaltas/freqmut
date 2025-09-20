// src/inference/cost_functions.rs

//! Cost functions for model optimization using `argmin`.

use crate::config::AppConfig;
use crate::data::ExperimentData;
use crate::interp_zero_alloc::{interp_zero_alloc, InterpMode};
use crate::models::parameters::*;
use argmin::core::{CostFunction, Error};
use ndarray::{Array1, Array2};

// =================================================================================
// Log Posterior
// =================================================================================
const KAP: f64 = 2.5;

/// Log-posterior of the gamma-Poisson distribution for read counts.
fn logpg(observed_reads: f64, expected_reads: f64) -> f64 {
    let expected_clamped = expected_reads.max(1.0);
    -(expected_clamped + observed_reads - 2.0 * (expected_clamped * observed_reads).sqrt()) / KAP
        - 0.5 * (4.0 * expected_clamped * KAP * std::f64::consts::PI).ln()
}

// =================================================================================
// Context and Cost Functions
// =================================================================================

/// Shared data context for all cost functions for a single lineage.
pub struct CostContext<'a> {
    pub lineage_idx: usize,
    pub config: &'a AppConfig,
    pub data: &'a ExperimentData,
    pub ee_matrix: &'a Array2<f64>,
    // Interpolation data pre-calculated for the current iteration
    pub afi: &'a Array1<f64>,
    pub ancf: &'a Array1<f64>,
    pub ancf_integral_t: &'a [f64],
    pub ancf_integral_v: &'a [f64],
    pub ancf_time_lookup_y: &'a [f64],
    pub ancf_time_lookup_x: &'a [f64],
}

// --- Neutral Model ---
pub struct NeutralCost<'a> {
    pub ctx: &'a CostContext<'a>,
}
impl CostFunction for NeutralCost<'_> {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, p: &Self::Param) -> Result<Self::Output, Error> {
        let r_ref = self.ctx.data.reads[[self.ctx.lineage_idx, 0]]
            / self.ctx.data.total_reads_per_timepoint[0];
        let x0 = conv_r0(p[0], r_ref, &self.ctx.config.bounds);

        let mut logp = 0.0;
        for (k, &tk) in self.ctx.data.t_points.iter().enumerate() {
            let expected_reads = self.ctx.data.total_reads_per_timepoint[k]
                * x0
                * self.ctx.ee_matrix[[self.ctx.data.t_points[0] as usize, tk as usize]];
            logp += logpg(self.ctx.data.reads[[self.ctx.lineage_idx, k]], expected_reads);
        }
        Ok(-logp)
    }
}

// --- Non-Ecology Model ---
pub struct NonEcologyCost<'a> {
    pub ctx: &'a CostContext<'a>,
}
impl CostFunction for NonEcologyCost<'_> {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, p: &Self::Param) -> Result<Self::Output, Error> {
        let r_ref = self.ctx.data.reads[[self.ctx.lineage_idx, 0]]
            / self.ctx.data.total_reads_per_timepoint[0];
        let x0 = conv_r0(p[0], r_ref, &self.ctx.config.bounds);
        let s_val = conv_s(p[1], &self.ctx.config.bounds);
        let tau = conv_tau(p[2], &self.ctx.config.bounds);

        let mut logp = 0.0;
        let interp_mode = InterpMode::Extrapolate;
        let eef_t1 = interp_zero_alloc(
            self.ctx.data.time_fine.as_slice().unwrap(),
            self.ctx.afi.as_slice().unwrap(),
            self.ctx.data.t_points[0] as f64,
            &interp_mode,
        );

        for (k, &tk) in self.ctx.data.t_points.iter().enumerate() {
            let rt_neutral = self.ctx.data.total_reads_per_timepoint[k]
                * x0
                * self.ctx.ee_matrix[[self.ctx.data.t_points[0] as usize, tk as usize]];

            let mut rt_mut = 0.0;
            if (tk as f64) >= tau {
                let eef_tk = interp_zero_alloc(
                    self.ctx.data.time_fine.as_slice().unwrap(),
                    self.ctx.afi.as_slice().unwrap(),
                    tk as f64,
                    &interp_mode,
                );
                rt_mut = 0.01 * x0 * self.ctx.data.total_reads_per_timepoint[k] * eef_tk / eef_t1
                    * (s_val * (tk as f64 - tau)).exp();
            }
            logp += logpg(self.ctx.data.reads[[self.ctx.lineage_idx, k]], rt_neutral + rt_mut);
        }
        Ok(-logp)
    }
}

// --- Ecology Model ---
pub struct EcologyCost<'a> {
    pub ctx: &'a CostContext<'a>,
}
impl CostFunction for EcologyCost<'_> {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, p: &Self::Param) -> Result<Self::Output, Error> {
        let r_ref = self.ctx.data.reads[[self.ctx.lineage_idx, 0]]
            / self.ctx.data.total_reads_per_timepoint[0];
        let x0 = conv_r0(p[0], r_ref, &self.ctx.config.bounds);
        let s_val = conv_s(p[1], &self.ctx.config.bounds);
        let tau = conv_tau(p[2], &self.ctx.config.bounds);
        let xc = conv_xc(p[3], &self.ctx.config.bounds);
        let m = conv_m(p[4], &self.ctx.config.bounds);

        let mut logp = 0.0;
        let interp_mode = InterpMode::Extrapolate;
        let tfine_slice = self.ctx.data.time_fine.as_slice().unwrap();
        let afi_slice = self.ctx.afi.as_slice().unwrap();
        let ancf_slice = self.ctx.ancf.as_slice().unwrap();

        let eef_t1 = interp_zero_alloc(
            tfine_slice,
            afi_slice,
            self.ctx.data.t_points[0] as f64,
            &interp_mode,
        );

        for (k, &tk) in self.ctx.data.t_points.iter().enumerate() {
            let rt_neutral = self.ctx.data.total_reads_per_timepoint[k]
                * x0
                * self.ctx.ee_matrix[[self.ctx.data.t_points[0] as usize, tk as usize]];

            let mut rt_mut = 0.0;
            if (tk as f64) >= tau {
                let eef_tk = interp_zero_alloc(tfine_slice, afi_slice, tk as f64, &interp_mode);
                rt_mut = 0.01 * x0 * self.ctx.data.total_reads_per_timepoint[k] * eef_tk / eef_t1
                    * (s_val * (tk as f64 - tau)).exp();

                let a_tau = interp_zero_alloc(tfine_slice, ancf_slice, tau, &interp_mode);
                if a_tau > xc {
                    let a_tk = interp_zero_alloc(tfine_slice, ancf_slice, tk as f64, &interp_mode);
                    
                    let tend = if a_tk < xc {
                        interp_zero_alloc(
                            self.ctx.ancf_time_lookup_y,
                            self.ctx.ancf_time_lookup_x,
                            xc,
                            &interp_mode
                        )
                    } else {
                        tk as f64
                    };

                    let aif_tend = interp_zero_alloc(self.ctx.ancf_integral_t, self.ctx.ancf_integral_v, tend, &interp_mode);
                    let aif_tau = interp_zero_alloc(self.ctx.ancf_integral_t, self.ctx.ancf_integral_v, tau, &interp_mode);
                    rt_mut *= (m * (aif_tend - aif_tau - xc * (tend - tau))).exp();
                }
            }
            logp += logpg(self.ctx.data.reads[[self.ctx.lineage_idx, k]], rt_neutral + rt_mut);
        }
        Ok(-logp)
    }
}