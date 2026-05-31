//! Root locus analysis: parameter sweep for variable gain.

use num_complex::Complex64;
use crate::TransferFunction;

/// A point on the root locus.
#[derive(Debug, Clone)]
pub struct RootLocusPoint {
    /// Gain value K
    pub gain: f64,
    /// Closed-loop poles at this gain
    pub poles: Vec<Complex64>,
}

/// Compute root locus for a given open-loop transfer function.
///
/// Sweeps gain K from `k_min` to `k_max` in `n_points` steps.
/// The closed-loop characteristic equation is: den(s) + K * num(s) = 0
pub fn root_locus(
    tf: &TransferFunction,
    k_min: f64,
    k_max: f64,
    n_points: usize,
) -> Vec<RootLocusPoint> {
    let mut points = Vec::with_capacity(n_points);
    for i in 0..n_points {
        let k = k_min + (k_max - k_min) * (i as f64) / ((n_points - 1).max(1) as f64);
        let poles = closed_loop_poles(tf, k);
        points.push(RootLocusPoint { gain: k, poles });
    }
    points
}

/// Find closed-loop poles for unity feedback with gain K.
pub fn closed_loop_poles(tf: &TransferFunction, k: f64) -> Vec<Complex64> {
    // Characteristic equation: den(s) + K * num(s) = 0
    let num = &tf.num;
    let den = &tf.den;

    // Compute den + K * num
    let max_len = den.len().max(num.len());
    let mut char_eq = vec![0.0; max_len];

    let offset_den = max_len - den.len();
    let offset_num = max_len - num.len();

    for (i, &c) in den.iter().enumerate() {
        char_eq[offset_den + i] += c;
    }
    for (i, &c) in num.iter().enumerate() {
        char_eq[offset_num + i] += k * c;
    }

    // Find roots
    TransferFunction::find_roots(&char_eq)
}

/// Find the critical gain (gain at which the root locus crosses the imaginary axis).
///
/// Uses the Routh-Hurwitz criterion to find the gain margin.
pub fn critical_gain(tf: &TransferFunction) -> Option<f64> {
    let num = &tf.num;
    let den = &tf.den;

    let max_len = den.len().max(num.len());
    let mut char_eq = vec![0.0; max_len];

    let offset_den = max_len - den.len();
    let offset_num = max_len - num.len();

    // We need to find K such that a row of the Routh array becomes zero.
    // For simple cases, we can sweep.
    for k_i in 0..10000 {
        let k = k_i as f64 * 0.01;
        char_eq.iter_mut().for_each(|c| *c = 0.0);
        for (i, &c) in den.iter().enumerate() {
            char_eq[offset_den + i] += c;
        }
        for (i, &c) in num.iter().enumerate() {
            char_eq[offset_num + i] += k * c;
        }

        if let Ok(rh) = crate::stability::routh_hurwitz(&char_eq) {
            if !rh.is_stable && k > 0.0 {
                return Some(k);
            }
        }
    }
    None
}

/// Find points where the root locus crosses the imaginary axis.
pub fn imaginary_axis_crossings(tf: &TransferFunction, n_points: usize) -> Vec<(f64, Complex64)> {
    let mut crossings = Vec::new();
    let points = root_locus(tf, 0.0, 100.0, n_points);

    for pt in &points {
        for pole in &pt.poles {
            if pole.re.abs() < 0.1 && pole.im.abs() > 0.01 {
                crossings.push((pt.gain, *pole));
            }
        }
    }

    crossings
}
