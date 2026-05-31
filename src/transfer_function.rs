//! Transfer function representation and analysis.

use num_complex::Complex64;
use num_traits::Zero;
use serde::{Deserialize, Serialize};

/// A rational transfer function H(s) = K * (s - z1)(s - z2)... / (s - p1)(s - p2)...
///
/// Stored in polynomial coefficient form: H(s) = num(s) / den(s)
/// where coefficients are in descending powers of s.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransferFunction {
    /// Numerator coefficients (descending powers of s)
    pub num: Vec<f64>,
    /// Denominator coefficients (descending powers of s)
    pub den: Vec<f64>,
}

impl TransferFunction {
    /// Create a new transfer function from numerator and denominator polynomial coefficients.
    /// Coefficients are in descending powers of s: [a_n, a_{n-1}, ..., a_0]
    pub fn new(num: Vec<f64>, den: Vec<f64>) -> Result<Self, String> {
        if den.is_empty() || den.iter().all(|&c| c.abs() < 1e-15) {
            return Err("Denominator cannot be zero".into());
        }
        // Trim leading near-zero coefficients from numerator
        let num = Self::trim_leading_zeros(&num);
        let den = Self::trim_leading_zeros(&den);
        Ok(Self { num, den })
    }

    /// Create a simple gain transfer function: H(s) = K
    pub fn gain(k: f64) -> Self {
        Self {
            num: vec![k],
            den: vec![1.0],
        }
    }

    /// Create a first-order transfer function: H(s) = K / (τs + 1)
    pub fn first_order(k: f64, tau: f64) -> Result<Self, String> {
        if tau.abs() < 1e-15 {
            return Err("Time constant tau cannot be zero".into());
        }
        Ok(Self {
            num: vec![k],
            den: vec![tau, 1.0],
        })
    }

    /// Create a second-order transfer function: H(s) = ωn² / (s² + 2ζωn·s + ωn²)
    pub fn second_order(wn: f64, zeta: f64) -> Result<Self, String> {
        if wn.abs() < 1e-15 {
            return Err("Natural frequency cannot be zero".into());
        }
        Ok(Self {
            num: vec![wn * wn],
            den: vec![1.0, 2.0 * zeta * wn, wn * wn],
        })
    }

    /// Evaluate the transfer function at a complex value s.
    pub fn evaluate(&self, s: Complex64) -> Complex64 {
        let num_val = Self::eval_poly(&self.num, s);
        let den_val = Self::eval_poly(&self.den, s);
        if den_val.norm() < 1e-15 {
            return Complex64::new(f64::INFINITY, 0.0);
        }
        num_val / den_val
    }

    /// Compute the DC gain (H(0)).
    pub fn dc_gain(&self) -> f64 {
        let s = Complex64::new(0.0, 0.0);
        self.evaluate(s).re
    }

    /// Compute the poles (roots of the denominator).
    pub fn poles(&self) -> Vec<Complex64> {
        Self::find_roots(&self.den)
    }

    /// Compute the zeros (roots of the numerator).
    pub fn zeros(&self) -> Vec<Complex64> {
        if self.num.is_empty() || (self.num.len() == 1 && self.num[0].abs() < 1e-15) {
            return vec![];
        }
        Self::find_roots(&self.num)
    }

    /// Compute the frequency response at angular frequency ω: H(jω).
    pub fn frequency_response(&self, omega: f64) -> Complex64 {
        let s = Complex64::new(0.0, omega);
        self.evaluate(s)
    }

    /// Series (cascade) connection: H(s) = H1(s) * H2(s)
    pub fn series(&self, other: &TransferFunction) -> TransferFunction {
        let num = Self::convolve(&self.num, &other.num);
        let den = Self::convolve(&self.den, &other.den);
        TransferFunction { num, den }
    }

    /// Parallel connection: H(s) = H1(s) + H2(s)
    pub fn parallel(&self, other: &TransferFunction) -> TransferFunction {
        let num = Self::poly_add(
            &Self::convolve(&self.num, &other.den),
            &Self::convolve(&self.den, &other.num),
        );
        let den = Self::convolve(&self.den, &other.den);
        TransferFunction { num, den }
    }

    /// Feedback connection: H_fb(s) = H(s) / (1 + H(s) * C(s))
    /// where C(s) is in the feedback path. Use gain(1.0) for unity feedback.
    pub fn feedback(&self, controller: &TransferFunction) -> TransferFunction {
        let open_loop = self.series(controller);
        let one_plus = open_loop.denom_plus_num();
        TransferFunction {
            num: self.num.clone(),
            den: one_plus,
        }
    }

    /// Returns the order of the transfer function (degree of denominator).
    pub fn order(&self) -> usize {
        if self.den.is_empty() {
            0
        } else {
            self.den.len() - 1
        }
    }

    /// Normalize so leading denominator coefficient is 1.
    pub fn normalize(&mut self) {
        if self.den.is_empty() {
            return;
        }
        let lead = self.den[0];
        if lead.abs() < 1e-15 {
            return;
        }
        for c in &mut self.num {
            *c /= lead;
        }
        for c in &mut self.den {
            *c /= lead;
        }
    }

    // --- Internal helpers ---

    fn eval_poly(coeffs: &[f64], s: Complex64) -> Complex64 {
        let mut result = Complex64::zero();
        for &c in coeffs {
            result = result * s + Complex64::new(c, 0.0);
        }
        result
    }

    fn trim_leading_zeros(v: &[f64]) -> Vec<f64> {
        let start = v.iter().position(|&c| c.abs() > 1e-15).unwrap_or(v.len().saturating_sub(1));
        if start >= v.len() {
            return vec![0.0];
        }
        v[start..].to_vec()
    }

    fn convolve(a: &[f64], b: &[f64]) -> Vec<f64> {
        if a.is_empty() || b.is_empty() {
            return vec![];
        }
        let n = a.len() + b.len() - 1;
        let mut result = vec![0.0; n];
        for i in 0..a.len() {
            for j in 0..b.len() {
                result[i + j] += a[i] * b[j];
            }
        }
        result
    }

    fn poly_add(a: &[f64], b: &[f64]) -> Vec<f64> {
        let n = a.len().max(b.len());
        let mut result = vec![0.0; n];
        let offset_a = n - a.len();
        let offset_b = n - b.len();
        for (i, &c) in a.iter().enumerate() {
            result[offset_a + i] += c;
        }
        for (i, &c) in b.iter().enumerate() {
            result[offset_b + i] += c;
        }
        result
    }

    /// Returns den(s) + num(s) as polynomial coefficients.
    fn denom_plus_num(&self) -> Vec<f64> {
        Self::poly_add(&self.den, &self.num)
    }

    /// Find roots of a polynomial using companion matrix eigenvalues (via Durand-Kerner).
    pub(crate) fn find_roots(coeffs: &[f64]) -> Vec<Complex64> {
        let c = Self::trim_leading_zeros(coeffs);
        let n = c.len();
        if n <= 1 {
            return vec![];
        }
        let degree = n - 1;
        if degree == 0 {
            return vec![];
        }
        // Normalize by leading coefficient
        let lead = c[0];
        if lead.abs() < 1e-15 {
            return vec![];
        }
        let normalized: Vec<f64> = c.iter().map(|&x| x / lead).collect();

        if degree == 1 {
            return vec![Complex64::new(-normalized[1], 0.0)];
        }

        if degree == 2 {
            let a = 1.0_f64;
            let b = normalized[1];
            let cc = normalized[2];
            let disc = b * b - 4.0 * a * cc;
            return if disc >= 0.0 {
                let sqrt_disc = disc.sqrt();
                vec![
                    Complex64::new((-b + sqrt_disc) / (2.0 * a), 0.0),
                    Complex64::new((-b - sqrt_disc) / (2.0 * a), 0.0),
                ]
            } else {
                let sqrt_disc = (-disc).sqrt();
                vec![
                    Complex64::new(-b / (2.0 * a), sqrt_disc / (2.0 * a)),
                    Complex64::new(-b / (2.0 * a), -sqrt_disc / (2.0 * a)),
                ]
            };
        }

        // Companion matrix approach using nalgebra
        Self::companion_eigenvalues(&normalized[1..])
    }

    /// Compute eigenvalues of the companion matrix.
    fn companion_eigenvalues(coeffs: &[f64]) -> Vec<Complex64> {
        let n = coeffs.len();
        if n == 0 {
            return vec![];
        }

        // Build companion matrix using nalgebra
        use nalgebra::DMatrix;
        let mut mat_data = vec![0.0_f64; n * n];

        // Sub-diagonal: ones
        for i in 1..n {
            mat_data[i * n + (i - 1)] = 1.0;
        }
        // Last column: -coeffs (reversed sign)
        for i in 0..n {
            mat_data[i * n + (n - 1)] = -coeffs[n - 1 - i];
        }

        let mat = DMatrix::from_row_slice(n, n, &mat_data);
        let eigen = mat.complex_eigenvalues();

        eigen.iter().map(|c| Complex64::new(c.re, c.im)).collect()
    }
}
