//! Bode plot analysis: magnitude, phase, gain and phase margins.

use crate::TransferFunction;

/// Bode plot data at a single frequency.
#[derive(Debug, Clone)]
pub struct BodePoint {
    /// Frequency (rad/s)
    pub frequency: f64,
    /// Magnitude in dB
    pub magnitude_db: f64,
    /// Phase in degrees
    pub phase_deg: f64,
}

/// Bode plot analysis results.
#[derive(Debug, Clone)]
pub struct BodeAnalysis {
    /// Bode data points
    pub points: Vec<BodePoint>,
    /// Gain margin in dB (None if no crossover)
    pub gain_margin_db: Option<f64>,
    /// Phase margin in degrees (None if no crossover)
    pub phase_margin_deg: Option<f64>,
    /// Gain crossover frequency (where magnitude = 0 dB)
    pub gain_crossover_freq: Option<f64>,
    /// Phase crossover frequency (where phase = -180°)
    pub phase_crossover_freq: Option<f64>,
}

/// Compute Bode plot data for a transfer function over a frequency range.
///
/// Frequencies are logarithmically spaced from `omega_min` to `omega_max`.
pub fn bode_plot(tf: &TransferFunction, omega_min: f64, omega_max: f64, n_points: usize) -> Vec<BodePoint> {
    let mut points = Vec::with_capacity(n_points);
    let log_min = omega_min.ln();
    let log_max = omega_max.ln();

    for i in 0..n_points {
        let log_omega = log_min + (log_max - log_min) * (i as f64) / ((n_points - 1).max(1) as f64);
        let omega = log_omega.exp();

        let h = tf.frequency_response(omega);
        let magnitude_db = 20.0 * h.norm().log10();
        let phase_deg = h.arg().to_degrees();

        points.push(BodePoint {
            frequency: omega,
            magnitude_db,
            phase_deg,
        });
    }

    points
}

/// Perform full Bode analysis including gain and phase margins.
pub fn bode_analysis(tf: &TransferFunction, omega_min: f64, omega_max: f64, n_points: usize) -> BodeAnalysis {
    let points = bode_plot(tf, omega_min, omega_max, n_points);

    // Find gain crossover frequency (where |H(jω)| = 0 dB)
    let mut gain_crossover_freq = None;
    for i in 1..points.len() {
        if points[i - 1].magnitude_db >= 0.0 && points[i].magnitude_db < 0.0 {
            // Linear interpolation
            let frac = points[i - 1].magnitude_db / (points[i - 1].magnitude_db - points[i].magnitude_db);
            let freq = points[i - 1].frequency * (points[i].frequency / points[i - 1].frequency).powf(frac);
            gain_crossover_freq = Some(freq);
            break;
        }
    }

    // Find phase crossover frequency (where phase crosses -180°)
    let mut phase_crossover_freq = None;
    for i in 1..points.len() {
        let p1 = points[i - 1].phase_deg;
        let p2 = points[i].phase_deg;
        if (p1 > -180.0 && p2 <= -180.0) || (p1 < -180.0 && p2 >= -180.0) {
            let frac = (p1 + 180.0) / (p1 - p2);
            let freq = points[i - 1].frequency * (points[i].frequency / points[i - 1].frequency).powf(frac);
            phase_crossover_freq = Some(freq);
            break;
        }
    }

    // Gain margin: -|H(jω_pc)| in dB at phase crossover
    let gain_margin_db = phase_crossover_freq.map(|f| {
        let h = tf.frequency_response(f);
        -20.0 * h.norm().log10()
    });

    // Phase margin: 180 + phase at gain crossover
    let phase_margin_deg = gain_crossover_freq.map(|f| {
        let h = tf.frequency_response(f);
        180.0 + h.arg().to_degrees()
    });

    BodeAnalysis {
        points,
        gain_margin_db,
        phase_margin_deg,
        gain_crossover_freq,
        phase_crossover_freq,
    }
}
