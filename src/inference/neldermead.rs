//! A custom, stack-allocated Nelder-Mead optimizer using const generics.

/// The result of a successful optimization run.
#[derive(Debug, Clone, Copy)]
pub struct OptimizationResult<const DIMS: usize> {
    pub best_param: [f64; DIMS],
    pub best_cost: f64,
}

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
    /// Creates a new solver with a given initial simplex.
    /// Panics if POINTS is not equal to DIMS + 1.
    pub fn new(initial_simplex: [[f64; DIMS]; POINTS]) -> Self {
        // We add a runtime check to enforce the rule.
        assert_eq!(POINTS, DIMS + 1, "The number of simplex points must be DIMS + 1.");
        Self {
            simplex: initial_simplex,
            costs: [f64::INFINITY; POINTS],
            alpha: 1.0,
            gamma: 2.0,
            rho: 0.5,
            sigma: 0.5,
        }
    }

    /// Runs the optimization loop.
    pub fn run<F>(&mut self, cost_fn: &F, max_iters: u64, tolerance: f64) -> OptimizationResult<DIMS>
    where
        F: Fn(&[f64]) -> Result<f64, argmin::core::Error>,
    {
        // Initial evaluation of all points in the simplex
        for i in 0..POINTS {
            self.costs[i] = cost_fn(&self.simplex[i]).unwrap();
        }

        for _iter in 0..max_iters {
            // 1. Order the vertices by cost
            let mut order: [(f64, usize); POINTS] = [(0.0, 0); POINTS];
            for i in 0..POINTS {
                order[i] = (self.costs[i], i);
            }
            order.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

            let best_idx = order[0].1;
            let second_worst_idx = order[POINTS-2].1;
            let worst_idx = order[POINTS-1].1;

            // Convergence check: if the costs are very close, stop.
            if self.costs[worst_idx] - self.costs[best_idx] < tolerance {
                break;
            }

            // 2. Calculate centroid of all points except the worst
            let mut centroid = [0.0; DIMS];
            for i in 0..=DIMS {
                if i != worst_idx {
                    for j in 0..DIMS {
                        centroid[j] += self.simplex[i][j] / (DIMS as f64);
                    }
                }
            }

            // 3. Reflection
            let mut reflected = [0.0; DIMS];
            for j in 0..DIMS {
                reflected[j] = centroid[j] + self.alpha * (centroid[j] - self.simplex[worst_idx][j]);
            }
            let reflected_cost = cost_fn(&reflected).unwrap();

            if self.costs[best_idx] <= reflected_cost && reflected_cost < self.costs[second_worst_idx] {
                // Case 1: Reflected point is better than second-worst, accept it.
                self.simplex[worst_idx] = reflected;
                self.costs[worst_idx] = reflected_cost;
            } else if reflected_cost < self.costs[best_idx] {
                // Case 2: Reflected point is the best so far, try expansion.
                let mut expanded = [0.0; DIMS];
                for j in 0..DIMS {
                    expanded[j] = centroid[j] + self.gamma * (reflected[j] - centroid[j]);
                }
                let expanded_cost = cost_fn(&expanded).unwrap();
                
                if expanded_cost < reflected_cost {
                    self.simplex[worst_idx] = expanded;
                    self.costs[worst_idx] = expanded_cost;
                } else {
                    self.simplex[worst_idx] = reflected;
                    self.costs[worst_idx] = reflected_cost;
                }
            } else {
                // Case 3: Reflected point is not good, try contraction.
                let mut contracted = [0.0; DIMS];
                for j in 0..DIMS {
                    contracted[j] = centroid[j] + self.rho * (self.simplex[worst_idx][j] - centroid[j]);
                }
                let contracted_cost = cost_fn(&contracted).unwrap();
                
                if contracted_cost < self.costs[worst_idx] {
                    self.simplex[worst_idx] = contracted;
                    self.costs[worst_idx] = contracted_cost;
                } else {
                    // Case 4: Contraction failed, shrink the simplex towards the best point.
                    for i in 0..=DIMS {
                        if i != best_idx {
                            for j in 0..DIMS {
                                self.simplex[i][j] = self.simplex[best_idx][j] + self.sigma * (self.simplex[i][j] - self.simplex[best_idx][j]);
                            }
                            self.costs[i] = cost_fn(&self.simplex[i]).unwrap();
                        }
                    }
                }
            }
        }

        // Find the best point in the final simplex to return
        let mut best_idx = 0;
        for i in 1..=DIMS {
            if self.costs[i] < self.costs[best_idx] {
                best_idx = i;
            }
        }
        
        OptimizationResult {
            best_param: self.simplex[best_idx],
            best_cost: self.costs[best_idx],
        }
    }
}