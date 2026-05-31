//! Stability analysis: Routh-Hurwitz criterion and Lyapunov stability.

use crate::state_space::StateSpace;

/// Result of Routh-Hurwitz stability analysis.
#[derive(Debug, Clone)]
pub struct RouthHurwitzResult {
    /// The Routh array rows
    pub routh_array: Vec<Vec<f64>>,
    /// Number of sign changes in the first column
    pub sign_changes: usize,
    /// Whether the system is stable (no sign changes, all positive coefficients)
    pub is_stable: bool,
    /// Number of right-half-plane roots
    pub rhp_roots: usize,
}

/// Build the Routh array and analyze stability for a polynomial.
///
/// Input: polynomial coefficients in descending powers of s.
/// Example: s³ + 2s² + 3s + 4 → [1, 2, 3, 4]
pub fn routh_hurwitz(coeffs: &[f64]) -> Result<RouthHurwitzResult, String> {
    if coeffs.len() < 2 {
        return Err("Need at least 2 coefficients for Routh-Hurwitz analysis".into());
    }
    if coeffs[0].abs() < 1e-15 {
        return Err("Leading coefficient cannot be zero".into());
    }

    let n = coeffs.len();
    let num_rows = n;
    let num_cols = (n + 1) / 2;

    let mut routh = vec![vec![0.0; num_cols]; num_rows];

    // First row: even-indexed coefficients
    for j in 0..num_cols {
        let idx = 2 * j;
        if idx < n {
            routh[0][j] = coeffs[idx];
        }
    }

    // Second row: odd-indexed coefficients
    for j in 0..num_cols {
        let idx = 2 * j + 1;
        if idx < n {
            routh[1][j] = coeffs[idx];
        }
    }

    // Fill remaining rows
    for i in 2..num_rows {
        let pivot = routh[i - 1][0];
        for j in 0..num_cols.saturating_sub(1) {
            if pivot.abs() < 1e-12 {
                routh[i][j] = 0.0;
            } else {
                routh[i][j] =
                    (pivot * routh[i - 2][j + 1] - routh[i - 2][0] * routh[i - 1][j + 1]) / pivot;
            }
        }

        // Handle zero first element: replace with small epsilon
        if routh[i][0].abs() < 1e-12 && i < num_rows - 1 {
            routh[i][0] = 1e-6;
        }

        // Handle entire row zero: use derivative of auxiliary polynomial
        let row_sum: f64 = routh[i].iter().map(|x| x.abs()).sum();
        if row_sum < 1e-10 && i < num_rows - 1 {
            // Replace with derivative of the auxiliary polynomial from row above
            for j in 0..num_cols {
                let power = (num_rows - 1 - (i - 1)) as i32 - (2 * j as i32);
                if power > 0 {
                    routh[i][j] = routh[i - 1][j] * power as f64;
                }
            }
        }
    }

    // Count sign changes in first column
    let first_col: Vec<f64> = routh
        .iter()
        .map(|r| r[0])
        .filter(|&v| v.abs() > 1e-12)
        .collect();
    let mut sign_changes = 0;
    for i in 1..first_col.len() {
        if first_col[i] * first_col[i - 1] < 0.0 {
            sign_changes += 1;
        }
    }

    let all_positive = coeffs.iter().all(|&c| c > -1e-10);

    Ok(RouthHurwitzResult {
        is_stable: sign_changes == 0 && all_positive,
        sign_changes,
        rhp_roots: sign_changes,
        routh_array: routh,
    })
}

/// Check if a polynomial represents a stable system (all roots in left half-plane).
pub fn is_polynomial_stable(coeffs: &[f64]) -> bool {
    if !coeffs.iter().all(|&c| c > -1e-12) {
        return false;
    }
    routh_hurwitz(coeffs).map(|r| r.is_stable).unwrap_or(false)
}

/// Compute the Hurwitz matrix for a characteristic polynomial.
pub fn hurwitz_matrix(coeffs: &[f64]) -> Vec<Vec<f64>> {
    let n = coeffs.len() - 1;
    if n == 0 {
        return vec![];
    }
    let a: Vec<f64> = coeffs.iter().map(|&c| c / coeffs[0]).collect();

    let mut h = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            let idx = 2 * (j + 1) - i;
            if idx > 0 && idx < a.len() {
                h[i][j] = a[idx];
            }
        }
    }
    h
}

/// Lyapunov stability analysis result.
#[derive(Debug)]
pub struct LyapunovResult {
    /// Whether the system is asymptotically stable
    pub is_stable: bool,
    /// The Lyapunov matrix P (solution to A'P + PA = -Q)
    pub p_matrix: Option<Vec<Vec<f64>>>,
}

/// Check Lyapunov stability by solving the continuous Lyapunov equation A'P + PA = -Q.
///
/// If Q is not provided, defaults to the identity matrix.
/// Returns the Lyapunov result with the solution P if the system is stable.
pub fn lyapunov_stability(ss: &StateSpace, q: Option<&[Vec<f64>]>) -> LyapunovResult {
    let n = ss.n_states();
    if n == 0 {
        return LyapunovResult {
            is_stable: true,
            p_matrix: Some(vec![]),
        };
    }

    let q_mat = if let Some(q) = q {
        let flat: Vec<f64> = q.iter().flat_map(|r| r.iter().copied()).collect();
        nalgebra::DMatrix::from_row_slice(n, n, &flat)
    } else {
        nalgebra::DMatrix::identity(n, n)
    };

    let a_flat: Vec<f64> = ss.a.iter().flat_map(|r| r.iter().copied()).collect();
    let a = nalgebra::DMatrix::from_row_slice(n, n, &a_flat);
    let a_t = a.transpose();

    // Build vectorized Lyapunov equation: [(I ⊗ A^T) + (A^T ⊗ I)] vec(P) = -vec(Q)
    let kron_size = n * n;
    let mut lhs = nalgebra::DMatrix::zeros(kron_size, kron_size);
    let mut rhs = nalgebra::DVector::zeros(kron_size);

    for i in 0..n {
        for j in 0..n {
            let row = i * n + j;
            rhs[row] = -q_mat[(i, j)];
            for k in 0..n {
                for l in 0..n {
                    let col = k * n + l;
                    if i == k {
                        lhs[(row, col)] += a_t[(j, l)];
                    }
                    if j == l {
                        lhs[(row, col)] += a_t[(i, k)];
                    }
                }
            }
        }
    }

    let p_vec = lhs.lu().solve(&rhs);

    match p_vec {
        Some(pv) => {
            let mut p_matrix = vec![vec![0.0; n]; n];
            for i in 0..n {
                for j in 0..n {
                    p_matrix[i][j] = pv[i * n + j];
                }
            }

            let p_flat: Vec<f64> = p_matrix.iter().flat_map(|r| r.iter().copied()).collect();
            let p_na = nalgebra::DMatrix::from_row_slice(n, n, &p_flat);
            let is_pd = p_na.cholesky().is_some();

            LyapunovResult {
                is_stable: is_pd,
                p_matrix: Some(p_matrix),
            }
        }
        None => LyapunovResult {
            is_stable: false,
            p_matrix: None,
        },
    }
}
