//! Observability and controllability Gramians.

use crate::state_space::StateSpace;
use nalgebra::DMatrix;

/// Controllability analysis result.
#[derive(Debug)]
pub struct ControllabilityResult {
    /// Whether the system is controllable
    pub is_controllable: bool,
    /// The controllability matrix [B, AB, A²B, ..., A^(n-1)B]
    pub controllability_matrix: Vec<Vec<f64>>,
    /// Rank of the controllability matrix
    pub rank: usize,
}

/// Observability analysis result.
#[derive(Debug)]
pub struct ObservabilityResult {
    /// Whether the system is observable
    pub is_observable: bool,
    /// The observability matrix [C; CA; CA²; ...; CA^(n-1)]
    pub observability_matrix: Vec<Vec<f64>>,
    /// Rank of the observability matrix
    pub rank: usize,
}

/// Compute the controllability matrix and check controllability.
///
/// A system is controllable if the controllability matrix has full rank.
pub fn controllability(ss: &StateSpace) -> ControllabilityResult {
    let n = ss.n_states();
    let m = ss.n_inputs();
    if n == 0 || m == 0 {
        return ControllabilityResult {
            is_controllable: false,
            controllability_matrix: vec![],
            rank: 0,
        };
    }

    let a_mat = matrix_to_nalgebra(&ss.a);
    let b_mat = matrix_to_nalgebra(&ss.b);

    // Controllability matrix: [B, AB, A²B, ..., A^(n-1)B]
    let mut col_blocks: Vec<DMatrix<f64>> = Vec::new();
    let mut a_power_b = b_mat.clone();
    col_blocks.push(a_power_b.clone());

    for _ in 1..n {
        a_power_b = &a_mat * &a_power_b;
        col_blocks.push(a_power_b.clone());
    }

    // Concatenate horizontally
    let c_matrix = concatenate_columns(&col_blocks, n, m * n);

    let rank = matrix_rank(&c_matrix);
    let c_vecs = nalgebra_to_vecs(&c_matrix);

    ControllabilityResult {
        is_controllable: rank == n,
        controllability_matrix: c_vecs,
        rank,
    }
}

/// Compute the observability matrix and check observability.
///
/// A system is observable if the observability matrix has full rank.
pub fn observability(ss: &StateSpace) -> ObservabilityResult {
    let n = ss.n_states();
    let p = ss.n_outputs();
    if n == 0 || p == 0 {
        return ObservabilityResult {
            is_observable: false,
            observability_matrix: vec![],
            rank: 0,
        };
    }

    let a_mat = matrix_to_nalgebra(&ss.a);
    let c_mat = matrix_to_nalgebra(&ss.c);

    // Observability matrix: [C; CA; CA²; ...; CA^(n-1)]
    let mut row_blocks: Vec<DMatrix<f64>> = Vec::new();
    let mut ca_power = c_mat.clone();
    row_blocks.push(ca_power.clone());

    for _ in 1..n {
        ca_power = &ca_power * &a_mat;
        row_blocks.push(ca_power.clone());
    }

    // Concatenate vertically
    let o_matrix = concatenate_rows(&row_blocks, p * n, n);

    let rank = matrix_rank(&o_matrix);
    let o_vecs = nalgebra_to_vecs(&o_matrix);

    ObservabilityResult {
        is_observable: rank == n,
        observability_matrix: o_vecs,
        rank,
    }
}

/// Compute the controllability Gramian: Wc = ∫₀^∞ e^(At) B B^T e^(A^T t) dt
///
/// Solved via the continuous Lyapunov equation: A Wc + Wc A^T = -B B^T
pub fn controllability_gramian(ss: &StateSpace) -> Option<Vec<Vec<f64>>> {
    let n = ss.n_states();
    if n == 0 {
        return Some(vec![]);
    }

    let a = matrix_to_nalgebra(&ss.a);
    let b = matrix_to_nalgebra(&ss.b);
    let bbt = &b * &b.transpose();

    solve_lyapunov(&a, &bbt, true)
}

/// Compute the observability Gramian: Wo = ∫₀^∞ e^(A^T t) C^T C e^(At) dt
///
/// Solved via the continuous Lyapunov equation: A^T Wo + Wo A = -C^T C
pub fn observability_gramian(ss: &StateSpace) -> Option<Vec<Vec<f64>>> {
    let n = ss.n_states();
    if n == 0 {
        return Some(vec![]);
    }

    let a = matrix_to_nalgebra(&ss.a);
    let c = matrix_to_nalgebra(&ss.c);
    let ctc = &c.transpose() * &c;

    solve_lyapunov(&a, &ctc, false)
}

/// Solve the continuous Lyapunov equation.
/// If transpose_a is true: A W + W A^T = -Q
/// Otherwise: A^T W + W A = -Q
fn solve_lyapunov(
    a: &DMatrix<f64>,
    q: &DMatrix<f64>,
    transpose_a: bool,
) -> Option<Vec<Vec<f64>>> {
    let n = a.nrows();
    let a_used = if transpose_a { a.clone() } else { a.transpose() };

    // Build vectorized system: [(I ⊗ A) + (A ⊗ I)] vec(W) = -vec(Q)
    let kron_size = n * n;
    let mut lhs = nalgebra::DMatrix::zeros(kron_size, kron_size);
    let mut rhs = nalgebra::DVector::zeros(kron_size);

    for i in 0..n {
        for j in 0..n {
            let row = i * n + j;
            rhs[row] = -q[(i, j)];
            for k in 0..n {
                for l in 0..n {
                    let col = k * n + l;
                    if i == k {
                        lhs[(row, col)] += a_used[(j, l)];
                    }
                    if j == l {
                        lhs[(row, col)] += a_used[(i, k)];
                    }
                }
            }
        }
    }

    let w_vec = lhs.lu().solve(&rhs)?;

    let mut w = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            w[i][j] = w_vec[i * n + j];
        }
    }
    Some(w)
}

fn matrix_to_nalgebra(m: &[Vec<f64>]) -> DMatrix<f64> {
    let rows = m.len();
    if rows == 0 {
        return DMatrix::zeros(0, 0);
    }
    let cols = m[0].len();
    let flat: Vec<f64> = m.iter().flat_map(|r| r.iter().copied()).collect();
    DMatrix::from_row_slice(rows, cols, &flat)
}

fn nalgebra_to_vecs(m: &DMatrix<f64>) -> Vec<Vec<f64>> {
    let mut result = vec![];
    for i in 0..m.nrows() {
        let mut row = vec![];
        for j in 0..m.ncols() {
            row.push(m[(i, j)]);
        }
        result.push(row);
    }
    result
}

fn concatenate_columns(blocks: &[DMatrix<f64>], n: usize, total_cols: usize) -> DMatrix<f64> {
    let mut data = vec![0.0; n * total_cols];
    let mut col_offset = 0;
    for block in blocks {
        for i in 0..n {
            for j in 0..block.ncols() {
                data[i * total_cols + col_offset + j] = block[(i, j)];
            }
        }
        col_offset += block.ncols();
    }
    DMatrix::from_row_slice(n, total_cols, &data)
}

fn concatenate_rows(blocks: &[DMatrix<f64>], total_rows: usize, n: usize) -> DMatrix<f64> {
    let mut data = vec![0.0; total_rows * n];
    let mut row_offset = 0;
    for block in blocks {
        for i in 0..block.nrows() {
            for j in 0..n {
                data[(row_offset + i) * n + j] = block[(i, j)];
            }
        }
        row_offset += block.nrows();
    }
    DMatrix::from_row_slice(total_rows, n, &data)
}

/// Estimate the rank of a matrix using SVD-like approach (singular value threshold).
fn matrix_rank(m: &DMatrix<f64>) -> usize {
    let svd = m.clone().svd(true, true);
    let tol = m.nrows().max(m.ncols()) as f64 * 1e-10;
    svd.singular_values.iter().filter(|&&s| s > tol).count()
}
