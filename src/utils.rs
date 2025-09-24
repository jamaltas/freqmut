// src/utils.rs

/// A simple linear interpolator.
/// Assumes x_pts is sorted.
#[derive(Clone)]
pub struct LinearInterpolator {
    x_pts: Vec<f64>,
    y_pts: Vec<f64>,
}

impl LinearInterpolator {
    pub fn new(x_pts: Vec<f64>, y_pts: Vec<f64>) -> Self {
        assert_eq!(x_pts.len(), y_pts.len(), "x and y points must have the same length");
        // For performance, we could also assert that x_pts is sorted.
        Self { x_pts, y_pts }
    }

    pub fn eval(&self, x: f64) -> f64 {
        if self.x_pts.is_empty() {
            return 0.0;
        }
        if x.is_nan() {
            return f64::NAN;
        }

        // Handle edge cases: x outside bounds (extrapolate flat)
        if x <= self.x_pts[0] {
            return self.y_pts[0];
        }
        if x >= self.x_pts[self.x_pts.len() - 1] {
            return self.y_pts[self.y_pts.len() - 1];
        }

        // Find the interval x is in using binary search
        let i = match self.x_pts.binary_search_by(|probe| probe.partial_cmp(&x).unwrap()) {
            Ok(i) => i, // Exact match
            Err(i) => i - 1, // i is the insertion point, so interval is [i-1, i]
        };

        // Linear interpolation formula
        let x0 = self.x_pts[i];
        let y0 = self.y_pts[i];
        let x1 = self.x_pts[i + 1];
        let y1 = self.y_pts[i + 1];

        if (x1 - x0).abs() < 1e-9 {
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

// Dummy interpolator for initialization
pub fn empty_interpolator() -> LinearInterpolator {
    LinearInterpolator::new(vec![], vec![])
}