// src/utils.rs

/// A simple linear interpolator.
/// Assumes x_pts is sorted.
#[derive(Clone)]
pub struct LinearInterpolator {
    x_pts: Vec<f64>,
    y_pts: Vec<f64>,
    uniform_metadata: Option<UniformGridMetadata>, // New field for optimization
}

// Metadata for uniformly spaced x points
#[derive(Clone, Copy)]
struct UniformGridMetadata {
    start: f64,
    step: f64,
    // The last index that can be the *start* of an interval for interpolation.
    // If len=5, indices are 0,1,2,3,4. Intervals are [0,1],[1,2],[2,3],[3,4].
    // So last valid start index is 3 (len-2).
    last_interval_start_idx: usize,
}

impl LinearInterpolator {
    pub fn new(x_pts: Vec<f64>, y_pts: Vec<f64>) -> Self {
        assert_eq!(x_pts.len(), y_pts.len(), "x and y points must have the same length");

        let uniform_metadata = if x_pts.len() >= 2 {
            let start = x_pts[0];
            let step = x_pts[1] - x_pts[0];
            let mut is_uniform = true;
            
            // If step is practically zero, all x_pts must be practically the same.
            // Use a relative epsilon for robustness.
            if step.abs() < f64::EPSILON * start.abs().max(x_pts[1].abs()).max(1.0) {
                for i in 2..x_pts.len() {
                    if (x_pts[i] - start).abs() > f64::EPSILON * x_pts[i].abs().max(start.abs()).max(1.0) {
                        is_uniform = false;
                        break;
                    }
                }
            } else { // Check for consistent non-zero step
                for i in 2..x_pts.len() {
                    let current_step = x_pts[i] - x_pts[i-1];
                    if (current_step - step).abs() > f64::EPSILON * step.abs().max(current_step.abs()).max(1.0) {
                        is_uniform = false;
                        break;
                    }
                }
            }

            if is_uniform {
                Some(UniformGridMetadata {
                    start,
                    step,
                    last_interval_start_idx: x_pts.len().saturating_sub(2),
                })
            } else {
                None
            }
        } else {
            None
        };

        Self { x_pts, y_pts, uniform_metadata }
    }

    pub fn eval(&self, x: f64) -> f64 {
        if self.x_pts.is_empty() {
            return 0.0;
        }
        if x.is_nan() {
            return f64::NAN;
        }

        let last_idx_val = self.x_pts.len() - 1; // Index of the last data point

        // Handle edge cases: x outside bounds (extrapolate flat)
        if x <= self.x_pts[0] {
            return self.y_pts[0];
        }
        if x >= self.x_pts[last_idx_val] {
            return self.y_pts[last_idx_val];
        }

        // If only one point exists (and not caught by above extrapolation checks, meaning x == x_pts[0])
        if self.x_pts.len() == 1 {
            return self.y_pts[0];
        }

        let i = if let Some(meta) = self.uniform_metadata {
            // Uniform grid calculation (O(1))
            let f_idx = (x - meta.start) / meta.step;
            let index = f_idx.floor() as usize;
            
            // Clamp index to a valid interval start [0, last_interval_start_idx]
            index.min(meta.last_interval_start_idx).max(0)
        } else {
            // Fallback to binary search (O(log N))
            // binary_search_by returns `Ok(idx)` for exact match, `Err(idx)` for insertion point.
            // We need the index `i` such that x_pts[i] <= x < x_pts[i+1].
            match self.x_pts.binary_search_by(|probe| probe.partial_cmp(&x).unwrap()) {
                Ok(i) => i.min(last_idx_val.saturating_sub(1)), // Ensure i+1 is valid
                Err(i) => (i - 1).min(last_idx_val.saturating_sub(1)).max(0), // Ensure i is valid and i+1 is valid
            }
        };

        // Linear interpolation formula
        let x0 = self.x_pts[i];
        let y0 = self.y_pts[i];
        let x1 = self.x_pts[i + 1];
        let y1 = self.y_pts[i + 1];

        // Robust floating point comparison for effectively zero denominator
        if (x1 - x0).abs() < f64::EPSILON * (x1.abs().max(x0.abs())).max(1.0) {
            return y0; // Points are essentially identical, return y0
        }

        y0 + (x - x0) * (y1 - y0) / (x1 - x0)
    }
}

/// A wrapper for LinearInterpolator that keeps track of the last found index
/// to speed up sequential lookups.
pub struct StatefulInterpolator<'a> {
    base: &'a LinearInterpolator,
    last_idx: usize,
}

impl<'a> StatefulInterpolator<'a> {
    pub fn new(interpolator: &'a LinearInterpolator) -> Self {
        Self {
            base: interpolator,
            last_idx: 0, // Start hint at the beginning
        }
    }

    /// Evaluates the interpolator at x, using and updating the internal hint.
    pub fn eval(&mut self, x: f64) -> f64 {
        let interp = self.base;

        if interp.x_pts.is_empty() {
            return 0.0;
        }
        if x.is_nan() {
            return f64::NAN;
        }

        let last_idx_val = interp.x_pts.len() - 1;

        // Handle edge cases: x outside bounds (extrapolate flat)
        if x <= interp.x_pts[0] {
            self.last_idx = 0; // Reset hint to start
            return interp.y_pts[0];
        }
        if x >= interp.x_pts[last_idx_val] {
            self.last_idx = if last_idx_val > 0 { last_idx_val - 1 } else { 0 }; // Reset hint to last valid interval start
            return interp.y_pts[last_idx_val];
        }

        // If only one point, and x is between bounds (i.e. x == x_pts[0]), return y_pts[0]
        if interp.x_pts.len() == 1 {
            self.last_idx = 0;
            return interp.y_pts[0];
        }
        
        // If the base interpolator uses a uniform grid, this is an O(1) lookup.
        // It's faster than any hint-based linear/binary search.
        if let Some(meta) = interp.uniform_metadata {
            let f_idx = (x - meta.start) / meta.step;
            let index = f_idx.floor() as usize;
            self.last_idx = index.min(meta.last_interval_start_idx).max(0);
            
            let i = self.last_idx; // Use the found index
            let x0 = interp.x_pts[i];
            let y0 = interp.y_pts[i];
            let x1 = interp.x_pts[i + 1];
            let y1 = interp.y_pts[i + 1];
            if (x1 - x0).abs() < f64::EPSILON * (x1.abs().max(x0.abs())).max(1.0) {
                return y0;
            }
            return y0 + (x - x0) * (y1 - y0) / (x1 - x0);
        }

        // Fallback for non-uniform grid, using hint-based search.
        let mut i = self.last_idx;
        // Ensure i is within valid bounds for an interval start [0, len-2]
        i = i.min(last_idx_val.saturating_sub(1)).max(0);

        // Check around the hint:
        // 1. Is x in the current interval? [i, i+1]
        // This is the fastest path if the hint is good.
        if x >= interp.x_pts[i] && x < interp.x_pts[i + 1] {
            // i is already correct, do nothing
        }
        // 2. Is x slightly ahead (in next few intervals)? Linear scan forward.
        // This is efficient for monotonically increasing x values.
        else if x >= interp.x_pts[i + 1] {
            let mut current_idx = i + 1;
            while current_idx < last_idx_val && x >= interp.x_pts[current_idx + 1] {
                current_idx += 1;
            }
            i = current_idx;
        }
        // 3. Is x slightly behind (in previous few intervals)? Linear scan backward.
        // For slightly decreasing x values.
        else if x < interp.x_pts[i] && i > 0 {
            let mut current_idx = i - 1;
            while current_idx > 0 && x < interp.x_pts[current_idx] {
                current_idx -= 1;
            }
            i = current_idx;
        }
        // 4. Hint is way off or x is far away, perform full binary search.
        // This path is taken if the sequential assumption is broken or the hint was bad.
        else {
            i = match interp.x_pts.binary_search_by(|probe| probe.partial_cmp(&x).unwrap()) {
                Ok(idx) => idx.min(last_idx_val.saturating_sub(1)),
                Err(idx) => (idx - 1).min(last_idx_val.saturating_sub(1)).max(0),
            };
        }
        
        self.last_idx = i; // Update hint for next call

        // Linear interpolation formula
        let x0 = interp.x_pts[i];
        let y0 = interp.y_pts[i];
        let x1 = interp.x_pts[i + 1];
        let y1 = interp.y_pts[i + 1];

        if (x1 - x0).abs() < f64::EPSILON * (x1.abs().max(x0.abs())).max(1.0) {
            return y0;
        }

        y0 + (x - x0) * (y1 - y0) / (x1 - x0)
    }
}

/// Calculates the median of a slice of f64.
pub fn median(data: &mut [f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    data.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mid = data.len() / 2;
    if data.len() % 2 == 0 {
        (data[mid - 1] + data[mid]) / 2.0
    } else {
        data[mid]
    }
}

// Dummy interpolator for initialization (no changes needed here)
pub fn empty_interpolator() -> LinearInterpolator {
    LinearInterpolator::new(vec![], vec![])
}