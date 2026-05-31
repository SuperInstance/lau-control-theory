//! Nyquist criterion: stability analysis from frequency response.

use num_complex::Complex64;
use crate::TransferFunction;

/// Nyquist plot data point.
#[derive(Debug, Clone)]
pub struct NyquistPoint {
    /// Frequency (rad/s)
    pub frequency: f64,
    /// Real part of H(jω)
    pub real: f64,
    /// Imaginary part of H(jω)
    pub imag: f64,
}

/// Nyquist analysis result.
#[derive(Debug, Clone)]
pub struct NyquistResult {
    /// Nyquist plot points
    pub points: Vec<NyquistPoint>,
    /// Number of encirclements of the point (-1, 0)
    pub encirclements: i32,
    /// Whether the closed-loop system is stable
    pub is_stable: bool,
    /// Number of open-loop right-half-plane poles
    pub open_loop_rhp_poles: usize,
}

/// Compute Nyquist plot data for a transfer function.
pub fn nyquist_plot(tf: &TransferFunction, omega_min: f64, omega_max: f64, n_points: usize) -> Vec<NyquistPoint> {
    let mut points = Vec::with_capacity(n_points);
    let log_min = omega_min.ln();
    let log_max = omega_max.ln();

    for i in 0..n_points {
        let log_omega = log_min + (log_max - log_min) * (i as f64) / ((n_points - 1).max(1) as f64);
        let omega = log_omega.exp();

        let h = tf.frequency_response(omega);
        points.push(NyquistPoint {
            frequency: omega,
            real: h.re,
            imag: h.im,
        });
    }

    points
}

/// Perform Nyquist stability analysis.
///
/// The Nyquist criterion: Z = N + P, where:
/// - Z = number of closed-loop RHP poles (must be 0 for stability)
/// - N = number of clockwise encirclements of (-1, 0)
/// - P = number of open-loop RHP poles
pub fn nyquist_analysis(
    tf: &TransferFunction,
    omega_min: f64,
    omega_max: f64,
    n_points: usize,
    open_loop_rhp_poles: usize,
) -> NyquistResult {
    let points = nyquist_plot(tf, omega_min, omega_max, n_points);

    // Count encirclements of (-1, 0) using winding number
    let encirclements = count_encirclements(&points, -1.0, 0.0);

    // Z = N + P (where N is clockwise = positive encirclements)
    let closed_loop_rhp_poles = encirclements.unsigned_abs() as usize + open_loop_rhp_poles;
    // For stability: need Z = 0, meaning N = -P
    let is_stable = if open_loop_rhp_poles == 0 {
        encirclements == 0
    } else {
        closed_loop_rhp_poles == 0
    };

    NyquistResult {
        points,
        encirclements,
        is_stable,
        open_loop_rhp_poles,
    }
}

/// Count the winding number of the Nyquist contour around a point (cx, cy).
///
/// Returns the net number of counterclockwise encirclements.
fn count_encirclements(points: &[NyquistPoint], cx: f64, cy: f64) -> i32 {
    let mut angle_sum = 0.0_f64;

    for i in 0..points.len() {
        let j = (i + 1) % points.len();
        let re1 = points[i].real - cx;
        let im1 = points[i].imag - cy;
        let re2 = points[j].real - cx;
        let im2 = points[j].imag - cy;

        let angle = Complex64::new(re1, im1).arg() - Complex64::new(re2, im2).arg();
        // Normalize to [-π, π]
        let angle = if angle > std::f64::consts::PI {
            angle - 2.0 * std::f64::consts::PI
        } else if angle < -std::f64::consts::PI {
            angle + 2.0 * std::f64::consts::PI
        } else {
            angle
        };

        angle_sum += angle;
    }

    // Winding number = angle_sum / (2π), rounded to nearest integer
    let winding = angle_sum / (2.0 * std::f64::consts::PI);
    winding.round() as i32
}

/// Check if the Nyquist plot passes through the critical point (-1, 0).
pub fn passes_through_critical(points: &[NyquistPoint], tolerance: f64) -> bool {
    points.iter().any(|p| {
        let dist = ((p.real + 1.0).powi(2) + p.imag.powi(2)).sqrt();
        dist < tolerance
    })
}
