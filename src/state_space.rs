//! State-space representation of linear time-invariant systems.
//!
//! ẋ = Ax + Bu, y = Cx + Du

use nalgebra::DVector;
use num_complex::Complex64;
use serde::{Deserialize, Serialize};

use crate::TransferFunction;

/// State-space representation of an LTI system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateSpace {
    /// State matrix (n x n)
    pub a: Vec<Vec<f64>>,
    /// Input matrix (n x m)
    pub b: Vec<Vec<f64>>,
    /// Output matrix (p x n)
    pub c: Vec<Vec<f64>>,
    /// Feedthrough matrix (p x m)
    pub d: Vec<Vec<f64>>,
}

impl StateSpace {
    /// Create a new state-space model from matrices.
    pub fn new(
        a: Vec<Vec<f64>>,
        b: Vec<Vec<f64>>,
        c: Vec<Vec<f64>>,
        d: Vec<Vec<f64>>,
    ) -> Result<Self, String> {
        let n = a.len();
        if n == 0 {
            return Err("State matrix A must be non-empty".into());
        }
        for row in &a {
            if row.len() != n {
                return Err("State matrix A must be square".into());
            }
        }
        // Validate b dimensions (n x m)
        let m = if !b.is_empty() { b[0].len() } else { 0 };
        if b.len() != n {
            return Err(format!("Input matrix B must have {} rows, got {}", n, b.len()));
        }
        // Validate c dimensions (p x n)
        let p = c.len();
        if p == 0 {
            return Err("Output matrix C must have at least one row".into());
        }
        for row in &c {
            if row.len() != n {
                return Err("Output matrix C must have n columns".into());
            }
        }
        // Validate d dimensions (p x m)
        if d.len() != p {
            return Err("Feedthrough matrix D must have p rows".into());
        }
        for row in &d {
            if row.len() != m {
                return Err("Feedthrough matrix D must have m columns".into());
            }
        }

        Ok(Self { a, b, c, d })
    }

    /// Number of states.
    pub fn n_states(&self) -> usize {
        self.a.len()
    }

    /// Number of inputs.
    pub fn n_inputs(&self) -> usize {
        if self.b.is_empty() { 0 } else { self.b[0].len() }
    }

    /// Number of outputs.
    pub fn n_outputs(&self) -> usize {
        self.c.len()
    }

    /// Compute eigenvalues of the A matrix (system poles).
    pub fn eigenvalues(&self) -> Vec<Complex64> {
        let n = self.n_states();
        let a_flat: Vec<f64> = self.a.iter().flat_map(|r| r.iter().copied()).collect();
        let mat = nalgebra::DMatrix::from_row_slice(n, n, &a_flat);
        let eigen = mat.complex_eigenvalues();
        eigen.iter().map(|c| Complex64::new(c.re, c.im)).collect()
    }

    /// Check if the system is stable (all eigenvalues have negative real parts).
    pub fn is_stable(&self) -> bool {
        self.eigenvalues().iter().all(|e| e.re < 0.0)
    }

    /// Convert to a transfer function (SISO only: single input, single output).
    pub fn to_transfer_function(&self) -> Result<TransferFunction, String> {
        if self.n_inputs() != 1 || self.n_outputs() != 1 {
            return Err("Transfer function conversion requires SISO system".into());
        }

        let n = self.n_states();
        let a_mat = nalgebra::DMatrix::from_row_slice(
            n, n, &self.a.iter().flat_map(|r| r.iter().copied()).collect::<Vec<_>>()
        );
        let b_vec = DVector::from_vec(
            self.b.iter().map(|r| r[0]).collect()
        );
        let c_vec = DVector::from_vec(
            self.c[0].clone()
        );
        let d_val = self.d[0][0];

        // Characteristic polynomial: det(sI - A)
        // Use companion form coefficients from eigenvalues
        let den = self.characteristic_polynomial();

        // For the numerator, we use the formula:
        // H(s) = C * adj(sI - A) * B + D * det(sI - A)
        // Simplified approach: compute using Leverrier's algorithm
        let (num, _) = self.leverrier(&a_mat, &b_vec, &c_vec, d_val, &den);

        TransferFunction::new(num, den)
    }

    /// Controllable canonical form for a SISO transfer function.
    pub fn controllable_canonical_form(tf: &TransferFunction) -> Result<Self, String> {
        if tf.num.len() > tf.den.len() {
            return Err("Transfer function must be proper (numerator degree <= denominator degree)".into());
        }

        let mut den = tf.den.clone();
        let lead = den[0];
        if lead.abs() < 1e-15 {
            return Err("Leading denominator coefficient cannot be zero".into());
        }
        den.iter_mut().for_each(|c| *c /= lead);

        let n = den.len() - 1;
        if n == 0 {
            return Ok(Self::new(
                vec![vec![]], // 0x0 A matrix
                vec![],
                vec![vec![]],
                vec![vec![tf.num.get(0).copied().unwrap_or(0.0) / lead]],
            ).unwrap_or_else(|_| Self::new(vec![vec![0.0]], vec![vec![1.0]], vec![vec![0.0]], vec![vec![tf.dc_gain()]]).unwrap()));
        }

        // Pad numerator to match denominator length
        let mut num = vec![0.0; den.len()];
        let offset = den.len() - tf.num.len();
        for (i, &c) in tf.num.iter().enumerate() {
            num[offset + i] = c / lead;
        }

        // A matrix: companion form
        let mut a = vec![vec![0.0; n]; n];
        // Sub-diagonal ones
        for i in 1..n {
            a[i][i - 1] = 1.0;
        }
        // Last row: -den coefficients (excluding leading 1)
        for i in 0..n {
            a[0][n - 1 - i] = -den[den.len() - 1 - i];
        }

        // B vector: [1, 0, ..., 0]^T
        let mut b = vec![vec![0.0]; n];
        b[0][0] = 1.0;

        // C: coefficients from numerator minus d*denominator
        let d_val = num[0];
        let mut c_row = vec![0.0; n];
        for i in 0..n {
            c_row[n - 1 - i] = num[num.len() - 1 - i] - d_val * den[den.len() - 1 - i];
        }

        // Wait, let me redo this properly.
        // In CCF: C = [bn-1, bn-2, ..., b0] where bi = num[i+1] - d*den[i+1] for i=0..n-1
        // Actually, the standard form is:
        // The numerator adjusted: num_adj[i] = num[i+1] - d_val * den[i+1] for i=0..n-1
        // C = [num_adj[n-1], num_adj[n-2], ..., num_adj[0]]
        let mut c_vec = vec![0.0; n];
        for i in 0..n {
            let idx = i + 1;
            let num_adj = if idx < num.len() { num[idx] } else { 0.0 }
                - d_val * if idx < den.len() { den[idx] } else { 0.0 };
            c_vec[n - 1 - i] = num_adj;
        }

        Self::new(a, b, vec![c_vec], vec![vec![d_val]])
    }

    /// Observable canonical form for a SISO transfer function.
    pub fn observable_canonical_form(tf: &TransferFunction) -> Result<Self, String> {
        let ccf = Self::controllable_canonical_form(tf)?;
        let n = ccf.n_states();
        if n == 0 {
            return Ok(ccf);
        }
        // Observable canonical form is the transpose of controllable canonical form
        let a_t = transpose_matrix(&ccf.a);
        let c_t = transpose_matrix(&ccf.b);
        let b_t = transpose_matrix(&ccf.c);

        Self::new(a_t, b_t, c_t, ccf.d)
    }

    /// Characteristic polynomial of A: det(sI - A) as descending coefficients.
    fn characteristic_polynomial(&self) -> Vec<f64> {
        let n = self.n_states();
        if n == 0 {
            return vec![1.0];
        }
        let a_flat: Vec<f64> = self.a.iter().flat_map(|r| r.iter().copied()).collect();
        let a_mat = nalgebra::DMatrix::from_row_slice(n, n, &a_flat);

        // Leverrier's method to compute characteristic polynomial

        let mut coeffs = vec![0.0; n + 1];
        coeffs[0] = 1.0; // leading coefficient

        let mut m = nalgebra::DMatrix::zeros(n, n);
        for k in 1..=n {
            if k == 1 {
                m = a_mat.clone();
            } else {
                // M_k = A * (M_{k-1} + c_{k-1} * I)
                let mut temp = m.clone();
                for i in 0..n {
                    temp[(i, i)] += coeffs[k - 1];
                }
                m = &a_mat * &temp;
            }
            // c_k = -trace(M_k) / k
            coeffs[k] = -m.trace() / (k as f64);
        }

        coeffs
    }

    /// Leverrier's algorithm for computing C*adj(sI-A)*B
    fn leverrier(
        &self,
        a: &nalgebra::DMatrix<f64>,
        b: &DVector<f64>,
        c: &DVector<f64>,
        d_val: f64,
        den: &[f64],
    ) -> (Vec<f64>, Vec<f64>) {
        let n = a.nrows();

        let mut char_coeffs = vec![0.0; n + 1];
        char_coeffs[0] = 1.0;

        let mut m = nalgebra::DMatrix::zeros(n, n);
        let mut num_coeffs = vec![0.0; n + 1];

        // d_val contribution to numerator
        num_coeffs[0] = d_val;

        for k in 1..=n {
            if k == 1 {
                m = a.clone();
            } else {
                let mut temp = m.clone();
                for i in 0..n {
                    temp[(i, i)] += char_coeffs[k - 1];
                }
                m = a * &temp;
            }
            char_coeffs[k] = -m.trace() / (k as f64);

            // Compute C * M_k * B for numerator coefficient
            let cmb = c.dot(&(&m * b));
            num_coeffs[k] = cmb;
        }

        // Adjust numerator: num(s) = C*adj(sI-A)*B + D*det(sI-A)
        // = sum_{k=0}^{n} (C*M_k*B + D*c_k) * s^{n-k}
        // where M_0 = 0, c_0 = 1
        for k in 1..=n {
            num_coeffs[k] += d_val * char_coeffs[k];
        }

        (num_coeffs, char_coeffs.to_vec())
    }

    /// Simulate the system for one time step using Euler integration.
    pub fn step(&self, x: &[f64], u: &[f64], dt: f64) -> (Vec<f64>, Vec<f64>) {
        let n = self.n_states();
        let p = self.n_outputs();

        // dx/dt = Ax + Bu
        let mut dx = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                dx[i] += self.a[i][j] * x[j];
            }
            for j in 0..u.len().min(self.n_inputs()) {
                dx[i] += self.b[i][j] * u[j];
            }
        }

        // x_new = x + dx * dt (Euler)
        let x_new: Vec<f64> = x.iter().zip(dx.iter()).map(|(&xi, &dxi)| xi + dxi * dt).collect();

        // y = Cx + Du
        let mut y = vec![0.0; p];
        for i in 0..p {
            for j in 0..n {
                y[i] += self.c[i][j] * x[j];
            }
            for j in 0..u.len().min(self.n_inputs()) {
                y[i] += self.d[i][j] * u[j];
            }
        }

        (x_new, y)
    }

    /// Simulate the system response to a step input.
    pub fn step_response(&self, t_end: f64, dt: f64) -> Vec<(f64, Vec<f64>)> {
        let n_steps = (t_end / dt).ceil() as usize;
        let mut results = Vec::with_capacity(n_steps + 1);
        let mut x = vec![0.0; self.n_states()];
        let u = vec![1.0; self.n_inputs()];

        for step in 0..=n_steps {
            let t = step as f64 * dt;
            let (_, y) = if step == 0 {
                (x.clone(), self.output(&x, &u))
            } else {
                let (new_x, y) = self.step(&x, &u, dt);
                x = new_x;
                (x.clone(), y)
            };
            results.push((t, y));
        }

        results
    }

    fn output(&self, x: &[f64], u: &[f64]) -> Vec<f64> {
        let p = self.n_outputs();
        let n = self.n_states();
        let mut y = vec![0.0; p];
        for i in 0..p {
            for j in 0..n {
                y[i] += self.c[i][j] * x[j];
            }
            for j in 0..u.len().min(self.n_inputs()) {
                y[i] += self.d[i][j] * u[j];
            }
        }
        y
    }
}

fn transpose_matrix(m: &[Vec<f64>]) -> Vec<Vec<f64>> {
    if m.is_empty() {
        return vec![];
    }
    let rows = m.len();
    let cols = m[0].len();
    let mut result = vec![vec![0.0; rows]; cols];
    for i in 0..rows {
        for j in 0..cols {
            result[j][i] = m[i][j];
        }
    }
    result
}
