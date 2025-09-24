//! A custom, stack-allocated Nelder-Mead optimizer using a builder pattern.
use std::fmt;

// --- Custom Error Type ---
/// Defines errors that can occur during solver configuration.
#[derive(Debug, Clone, PartialEq)]
pub enum NelderMeadError {
    InvalidParameter(String),
}

impl fmt::Display for NelderMeadError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            NelderMeadError::InvalidParameter(msg) => write!(f, "Invalid Nelder-Mead parameter: {}", msg),
        }
    }
}

impl std::error::Error for NelderMeadError {}


// --- Optimization Result ---
/// The result of a successful optimization run.
#[derive(Debug, Clone, Copy)]
pub struct OptimizationResult<const DIMS: usize> {
    pub best_param: [f64; DIMS],
    pub best_cost: f64,
}

// --- Nelder-Mead Solver ---
/// A stack-allocated Nelder-Mead solver.
/// DIMS: The number of dimensions of the optimization problem.
/// POINTS: The number of vertices in the simplex (must be DIMS + 1).
pub struct NelderMead<const DIMS: usize, const POINTS: usize> {
    // The simplex now uses the POINTS generic directly.
    simplex: [[f64; DIMS]; POINTS],
    // The cost associated with each point in the simplex.
    costs: [f64; POINTS],
    // Algorithm coefficients
    alpha: f64, // reflection
    gamma: f64, // expansion
    rho: f64,   // contraction
    sigma: f64, // shrink
}

impl<const DIMS: usize, const POINTS: usize> NelderMead<DIMS, POINTS> {
    /// Creates a new solver with a given initial simplex and default coefficients.
    /// Panics if POINTS is not equal to DIMS + 1.
    pub fn new(initial_simplex: [[f64; DIMS]; POINTS]) -> Self {
        // We add a runtime check to enforce the rule.
        assert_eq!(POINTS, DIMS + 1, "The number of simplex points must be DIMS + 1.");
        Self {
            simplex: initial_simplex,
            costs: [f64::INFINITY; POINTS],
            // Standard, proven default coefficients
            alpha: 1.0,
            gamma: 2.0,
            rho: 0.5,
            sigma: 0.5,
        }
    }
    
    // --- Builder Methods ---

    /// Sets the reflection coefficient (alpha). Must be positive.
    pub fn with_alpha(mut self, alpha: f64) -> Result<Self, NelderMeadError> {
        if alpha <= 0.0 {
            return Err(NelderMeadError::InvalidParameter(
                "Reflection coefficient (alpha) must be positive.".to_string()
            ));
        }
        self.alpha = alpha;
        Ok(self)
    }

    /// Sets the expansion coefficient (gamma). Must be greater than 1.0.
    pub fn with_gamma(mut self, gamma: f64) -> Result<Self, NelderMeadError> {
        if gamma <= 1.0 {
            return Err(NelderMeadError::InvalidParameter(
                "Expansion coefficient (gamma) must be greater than 1.0.".to_string()
            ));
        }
        self.gamma = gamma;
        Ok(self)
    }

    /// Sets the contraction coefficient (rho). Must be in the range (0.0, 1.0).
    pub fn with_rho(mut self, rho: f64) -> Result<Self, NelderMeadError> {
        if !(0.0..1.0).contains(&rho) {
            return Err(NelderMeadError::InvalidParameter(
                "Contraction coefficient (rho) must be in the range (0.0, 1.0).".to_string()
            ));
        }
        self.rho = rho;
        Ok(self)
    }

    /// Sets the shrink coefficient (sigma). Must be in the range (0.0, 1.0).
    pub fn with_sigma(mut self, sigma: f64) -> Result<Self, NelderMeadError> {
        if !(0.0..1.0).contains(&sigma) {
            return Err(NelderMeadError::InvalidParameter(
                "Shrink coefficient (sigma) must be in the range (0.0, 1.0).".to_string()
            ));
        }
        self.sigma = sigma;
        Ok(self)
    }

    // --- Core Logic (including fixes from previous review) ---

    /// Helper method to perform the shrink operation.
    fn shrink<F>(&mut self, best_idx: usize, cost_fn: &F)
    where
        F: Fn(&[f64; DIMS]) -> Result<f64, argmin::core::Error>,
    {
        for i in 0..POINTS {
            if i != best_idx {
                for j in 0..DIMS {
                    self.simplex[i][j] = self.simplex[best_idx][j] + self.sigma * (self.simplex[i][j] - self.simplex[best_idx][j]);
                }
                self.costs[i] = cost_fn(&self.simplex[i]).unwrap();
            }
        }
    }

    /// Runs the optimization loop.
    pub fn run<F>(&mut self, cost_fn: F, max_iters: u64, tolerance: f64) -> OptimizationResult<DIMS>
    where
        F: Fn(&[f64; DIMS]) -> Result<f64, argmin::core::Error>,
    {
        for i in 0..POINTS {
            self.costs[i] = cost_fn(&self.simplex[i]).unwrap();
        }

        for _iter in 0..max_iters {
            let mut order: [(f64, usize); POINTS] = [(0.0, 0); POINTS];
            for i in 0..POINTS { order[i] = (self.costs[i], i); }
            order.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

            let best_idx = order[0].1;
            let second_worst_idx = order[POINTS - 2].1;
            let worst_idx = order[POINTS - 1].1;
            
            if (self.costs[worst_idx] - self.costs[best_idx]).abs() < tolerance { break; }

            let mut centroid = [0.0; DIMS];
            for i in 0..POINTS {
                if i != worst_idx {
                    for j in 0..DIMS { centroid[j] += self.simplex[i][j]; }
                }
            }
            for j in 0..DIMS { centroid[j] /= DIMS as f64; }

            let mut reflected = [0.0; DIMS];
            for j in 0..DIMS { reflected[j] = centroid[j] + self.alpha * (centroid[j] - self.simplex[worst_idx][j]); }
            let reflected_cost = cost_fn(&reflected).unwrap();
            
            if self.costs[best_idx] <= reflected_cost && reflected_cost < self.costs[second_worst_idx] {
                self.simplex[worst_idx] = reflected;
                self.costs[worst_idx] = reflected_cost;
            } else if reflected_cost < self.costs[best_idx] {
                let mut expanded = [0.0; DIMS];
                for j in 0..DIMS { expanded[j] = centroid[j] + self.gamma * (reflected[j] - centroid[j]); }
                let expanded_cost = cost_fn(&expanded).unwrap();
                
                if expanded_cost < reflected_cost {
                    self.simplex[worst_idx] = expanded;
                    self.costs[worst_idx] = expanded_cost;
                } else {
                    self.simplex[worst_idx] = reflected;
                    self.costs[worst_idx] = reflected_cost;
                }
            } else {
                let mut contracted = [0.0; DIMS];
                if reflected_cost < self.costs[worst_idx] {
                    for j in 0..DIMS { contracted[j] = centroid[j] + self.rho * (reflected[j] - centroid[j]); }
                    let contracted_cost = cost_fn(&contracted).unwrap();
                    if contracted_cost <= reflected_cost {
                        self.simplex[worst_idx] = contracted;
                        self.costs[worst_idx] = contracted_cost;
                    } else { self.shrink(best_idx, &cost_fn); }
                } else {
                    for j in 0..DIMS { contracted[j] = centroid[j] + self.rho * (self.simplex[worst_idx][j] - centroid[j]); }
                    let contracted_cost = cost_fn(&contracted).unwrap();
                    if contracted_cost < self.costs[worst_idx] {
                        self.simplex[worst_idx] = contracted;
                        self.costs[worst_idx] = contracted_cost;
                    } else { self.shrink(best_idx, &cost_fn); }
                }
            }
        }

        let mut best_idx = 0;
        for i in 1..POINTS {
            if self.costs[i] < self.costs[best_idx] { best_idx = i; }
        }
        
        OptimizationResult {
            best_param: self.simplex[best_idx],
            best_cost: self.costs[best_idx],
        }
    }
}

