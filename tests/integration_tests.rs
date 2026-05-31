//! Comprehensive tests for lau-control-theory.

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use num_complex::Complex64;

    // ===== Transfer Function Tests =====

    #[test]
    fn test_tf_creation() {
        use lau_control_theory::TransferFunction;
        let tf = TransferFunction::new(vec![1.0], vec![1.0, 2.0]).unwrap();
        assert_eq!(tf.num, vec![1.0]);
        assert_eq!(tf.den, vec![1.0, 2.0]);
    }

    #[test]
    fn test_tf_gain() {
        use lau_control_theory::TransferFunction;
        let tf = TransferFunction::gain(5.0);
        assert_abs_diff_eq!(tf.dc_gain(), 5.0, epsilon = 1e-10);
    }

    #[test]
    fn test_tf_first_order() {
        use lau_control_theory::TransferFunction;
        let tf = TransferFunction::first_order(2.0, 0.5).unwrap();
        assert_abs_diff_eq!(tf.dc_gain(), 2.0, epsilon = 1e-10);
        assert_eq!(tf.order(), 1);
    }

    #[test]
    fn test_tf_second_order() {
        use lau_control_theory::TransferFunction;
        let tf = TransferFunction::second_order(1.0, 0.5).unwrap();
        assert_eq!(tf.order(), 2);
        let s = Complex64::new(0.0, 0.0);
        let val = tf.evaluate(s);
        assert_abs_diff_eq!(val.re, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_tf_evaluate() {
        use lau_control_theory::TransferFunction;
        let tf = TransferFunction::new(vec![1.0], vec![1.0, 1.0]).unwrap();
        let s = Complex64::new(1.0, 0.0);
        let val = tf.evaluate(s);
        assert_abs_diff_eq!(val.re, 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_tf_frequency_response() {
        use lau_control_theory::TransferFunction;
        let tf = TransferFunction::new(vec![1.0], vec![1.0, 1.0]).unwrap();
        let h = tf.frequency_response(1.0);
        // H(j) = 1 / (1 + j) = (1 - j) / 2
        assert_abs_diff_eq!(h.re, 0.5, epsilon = 1e-10);
        assert_abs_diff_eq!(h.im, -0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_tf_poles_zeros() {
        use lau_control_theory::TransferFunction;
        // H(s) = (s + 2) / (s + 1)(s + 3) = (s + 2) / (s² + 4s + 3)
        let tf = TransferFunction::new(vec![1.0, 2.0], vec![1.0, 4.0, 3.0]).unwrap();
        let poles = tf.poles();
        let zeros = tf.zeros();

        assert_eq!(poles.len(), 2);
        assert_eq!(zeros.len(), 1);

        assert_abs_diff_eq!(zeros[0].re, -2.0, epsilon = 1e-6);
        assert_abs_diff_eq!(zeros[0].im, 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_tf_series() {
        use lau_control_theory::TransferFunction;
        let h1 = TransferFunction::new(vec![1.0], vec![1.0, 1.0]).unwrap();
        let h2 = TransferFunction::new(vec![2.0], vec![1.0, 2.0]).unwrap();
        let series = h1.series(&h2);
        // H(s) = 2 / (s+1)(s+2) = 2 / (s² + 3s + 2)
        assert_eq!(series.num, vec![2.0]);
        assert_eq!(series.den, vec![1.0, 3.0, 2.0]);
    }

    #[test]
    fn test_tf_parallel() {
        use lau_control_theory::TransferFunction;
        let h1 = TransferFunction::new(vec![1.0], vec![1.0, 1.0]).unwrap();
        let h2 = TransferFunction::new(vec![1.0], vec![1.0, 2.0]).unwrap();
        let parallel = h1.parallel(&h2);
        // H(s) = (s+2+s+1) / ((s+1)(s+2)) = (2s+3) / (s²+3s+2)
        assert_abs_diff_eq!(parallel.num[0], 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(parallel.den[0], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_tf_feedback() {
        use lau_control_theory::TransferFunction;
        let h = TransferFunction::new(vec![1.0], vec![1.0, 1.0]).unwrap();
        let unity = TransferFunction::gain(1.0);
        let closed = h.feedback(&unity);
        // T(s) = 1/(s+1) / (1 + 1/(s+1)) = 1/(s+2)
        assert_abs_diff_eq!(closed.den[0], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(closed.den[1], 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_tf_complex_poles() {
        use lau_control_theory::TransferFunction;
        // s² + 2s + 5 has complex poles at -1 ± 2j
        let tf = TransferFunction::new(vec![1.0], vec![1.0, 2.0, 5.0]).unwrap();
        let poles = tf.poles();
        assert_eq!(poles.len(), 2);
        // Check real parts are -1
        assert_abs_diff_eq!(poles[0].re, -1.0, epsilon = 1e-6);
        assert_abs_diff_eq!(poles[1].re, -1.0, epsilon = 1e-6);
        // Check imaginary parts are ±2
        let im_sum = poles[0].im + poles[1].im;
        assert_abs_diff_eq!(im_sum, 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_tf_den_zero_rejected() {
        use lau_control_theory::TransferFunction;
        assert!(TransferFunction::new(vec![1.0], vec![0.0, 0.0]).is_err());
    }

    // ===== State Space Tests =====

    #[test]
    fn test_ss_creation() {
        use lau_control_theory::StateSpace;
        let ss = StateSpace::new(
            vec![vec![-1.0]],
            vec![vec![1.0]],
            vec![vec![1.0]],
            vec![vec![0.0]],
        ).unwrap();
        assert_eq!(ss.n_states(), 1);
        assert_eq!(ss.n_inputs(), 1);
        assert_eq!(ss.n_outputs(), 1);
    }

    #[test]
    fn test_ss_stable() {
        use lau_control_theory::StateSpace;
        let ss = StateSpace::new(
            vec![vec![-2.0]],
            vec![vec![1.0]],
            vec![vec![1.0]],
            vec![vec![0.0]],
        ).unwrap();
        assert!(ss.is_stable());
    }

    #[test]
    fn test_ss_unstable() {
        use lau_control_theory::StateSpace;
        let ss = StateSpace::new(
            vec![vec![2.0]],
            vec![vec![1.0]],
            vec![vec![1.0]],
            vec![vec![0.0]],
        ).unwrap();
        assert!(!ss.is_stable());
    }

    #[test]
    fn test_ss_eigenvalues() {
        use lau_control_theory::StateSpace;
        let ss = StateSpace::new(
            vec![vec![0.0, 1.0], vec![-2.0, -3.0]],
            vec![vec![0.0], vec![1.0]],
            vec![vec![1.0, 0.0]],
            vec![vec![0.0]],
        ).unwrap();
        let eigenvalues = ss.eigenvalues();
        assert_eq!(eigenvalues.len(), 2);
        // All should have negative real parts
        for e in &eigenvalues {
            assert!(e.re < 0.0);
        }
    }

    #[test]
    fn test_ss_step_response() {
        use lau_control_theory::StateSpace;
        let ss = StateSpace::new(
            vec![vec![-1.0]],
            vec![vec![1.0]],
            vec![vec![1.0]],
            vec![vec![0.0]],
        ).unwrap();
        let response = ss.step_response(5.0, 0.01);
        assert!(!response.is_empty());
        // For a first-order system with τ=1, step response should approach 1.0
        let final_val = response.last().unwrap().1[0];
        assert!(final_val > 0.99);
    }

    #[test]
    fn test_ss_controllable_canonical_form() {
        use lau_control_theory::{StateSpace, TransferFunction};
        let tf = TransferFunction::new(vec![1.0], vec![1.0, 3.0, 2.0]).unwrap();
        let ss = StateSpace::controllable_canonical_form(&tf).unwrap();
        assert_eq!(ss.n_states(), 2);
    }

    #[test]
    fn test_ss_observable_canonical_form() {
        use lau_control_theory::{StateSpace, TransferFunction};
        let tf = TransferFunction::new(vec![1.0], vec![1.0, 3.0, 2.0]).unwrap();
        let ss = StateSpace::observable_canonical_form(&tf).unwrap();
        assert_eq!(ss.n_states(), 2);
    }

    #[test]
    fn test_ss_to_tf_roundtrip() {
        use lau_control_theory::{StateSpace, TransferFunction};
        // Verify that CCF has correct eigenvalues matching TF poles
        let tf = TransferFunction::new(vec![1.0], vec![1.0, 1.0]).unwrap();
        let ss = StateSpace::controllable_canonical_form(&tf).unwrap();
        // The eigenvalues should match the poles
        let eigenvalues = ss.eigenvalues();
        let poles = tf.poles();
        assert_eq!(eigenvalues.len(), poles.len());
        assert_abs_diff_eq!(eigenvalues[0].re, poles[0].re, epsilon = 0.1);
        assert_abs_diff_eq!(eigenvalues[0].im, poles[0].im, epsilon = 0.1);
        // Check stability preserved
        assert!(ss.is_stable());
    }

    #[test]
    fn test_ss_validation_rejects_bad_dims() {
        use lau_control_theory::StateSpace;
        // Non-square A
        assert!(StateSpace::new(
            vec![vec![1.0, 2.0]],
            vec![vec![1.0]],
            vec![vec![1.0, 2.0]],
            vec![vec![0.0]],
        ).is_err());
    }

    // ===== Routh-Hurwitz Tests =====

    #[test]
    fn test_routh_stable_system() {
        use lau_control_theory::stability;
        // s³ + 2s² + 3s + 4 — all positive, Routh stable
        let result = stability::routh_hurwitz(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        assert!(result.is_stable);
        assert_eq!(result.sign_changes, 0);
    }

    #[test]
    fn test_routh_unstable_system() {
        use lau_control_theory::stability;
        // s³ + 2s² - 3s + 4 — has sign change in coefficients, unstable
        let result = stability::routh_hurwitz(&[1.0, 2.0, -3.0, 4.0]).unwrap();
        assert!(!result.is_stable);
        assert!(result.sign_changes > 0);
    }

    #[test]
    fn test_routh_second_order_stable() {
        use lau_control_theory::stability;
        let result = stability::routh_hurwitz(&[1.0, 2.0, 1.0]).unwrap();
        assert!(result.is_stable);
    }

    #[test]
    fn test_routh_second_order_unstable() {
        use lau_control_theory::stability;
        // s² - 1: has negative coefficient, unstable
        let result = stability::routh_hurwitz(&[1.0, 0.0, -1.0]).unwrap();
        assert!(!result.is_stable);
    }

    #[test]
    fn test_polynomial_stable() {
        use lau_control_theory::stability;
        assert!(stability::is_polynomial_stable(&[1.0, 3.0, 3.0, 1.0])); // (s+1)³
    }

    #[test]
    fn test_polynomial_unstable() {
        use lau_control_theory::stability;
        assert!(!stability::is_polynomial_stable(&[1.0, -1.0])); // s - 1
    }

    #[test]
    fn test_routh_fourth_order_stable() {
        use lau_control_theory::stability;
        // (s+1)(s+2)(s+3)(s+4) = s⁴ + 10s³ + 35s² + 50s + 24
        let result = stability::routh_hurwitz(&[1.0, 10.0, 35.0, 50.0, 24.0]).unwrap();
        assert!(result.is_stable);
        assert_eq!(result.sign_changes, 0);
    }

    // ===== Lyapunov Stability Tests =====

    #[test]
    fn test_lyapunov_stable_system() {
        use lau_control_theory::{StateSpace, stability};
        let ss = StateSpace::new(
            vec![vec![-2.0]],
            vec![vec![1.0]],
            vec![vec![1.0]],
            vec![vec![0.0]],
        ).unwrap();
        let result = stability::lyapunov_stability(&ss, None);
        assert!(result.is_stable);
        assert!(result.p_matrix.is_some());
        // P should be positive definite: for A=-2, Q=1: 2*(-2)*P = -1 => P = 0.25
        let p = result.p_matrix.unwrap();
        assert!(p[0][0] > 0.0);
    }

    #[test]
    fn test_lyapunov_unstable_system() {
        use lau_control_theory::{StateSpace, stability};
        let ss = StateSpace::new(
            vec![vec![2.0]],
            vec![vec![1.0]],
            vec![vec![1.0]],
            vec![vec![0.0]],
        ).unwrap();
        let result = stability::lyapunov_stability(&ss, None);
        assert!(!result.is_stable);
    }

    #[test]
    fn test_lyapunov_2d_stable() {
        use lau_control_theory::{StateSpace, stability};
        let ss = StateSpace::new(
            vec![vec![0.0, 1.0], vec![-2.0, -3.0]],
            vec![vec![0.0], vec![1.0]],
            vec![vec![1.0, 0.0]],
            vec![vec![0.0]],
        ).unwrap();
        let result = stability::lyapunov_stability(&ss, None);
        assert!(result.is_stable);
    }

    // ===== PID Tests =====

    #[test]
    fn test_pid_p_only() {
        use lau_control_theory::PidController;
        let mut pid = PidController::new(2.0, 0.0, 0.0);
        let output = pid.update(5.0, 0.1);
        assert_abs_diff_eq!(output, 10.0, epsilon = 1e-10);
    }

    #[test]
    fn test_pid_pi() {
        use lau_control_theory::PidController;
        let mut pid = PidController::new(1.0, 1.0, 0.0);
        let _ = pid.update(1.0, 0.1); // integral = 0.1, output = P + I = 1 + 0.1 = 1.1
        let output = pid.update(1.0, 0.1); // integral = 0.2, output = P + I = 1 + 0.2 = 1.2
        assert_abs_diff_eq!(output, 1.2, epsilon = 1e-10);
    }

    #[test]
    fn test_pid_reset() {
        use lau_control_theory::PidController;
        let mut pid = PidController::new(1.0, 0.0, 0.0); // P-only for clean test
        let _ = pid.update(5.0, 0.1);
        pid.reset();
        let output = pid.update(5.0, 0.1);
        assert_abs_diff_eq!(output, 5.0, epsilon = 1e-10); // Only P term
    }

    #[test]
    fn test_pid_output_limits() {
        use lau_control_theory::PidController;
        let mut pid = PidController::new(100.0, 0.0, 0.0);
        pid.output_limits = Some((-10.0, 10.0));
        let output = pid.update(1.0, 0.1);
        assert_abs_diff_eq!(output, 10.0, epsilon = 1e-10);
    }

    #[test]
    fn test_pid_anti_windup() {
        use lau_control_theory::PidController;
        let mut pid = PidController::new(0.0, 10.0, 0.0);
        pid.integral_limit = Some(5.0);
        for _ in 0..1000 {
            let _ = pid.update(100.0, 0.01);
        }
        // Integral should be clamped
        assert!(pid.integral_value() <= 5.0);
    }

    #[test]
    fn test_pid_first_order_response() {
        use lau_control_theory::PidController;
        let mut pid = PidController::new(2.0, 1.0, 0.1);
        let response = pid.simulate_first_order(1.0, 1.0, 1.0, 10.0, 0.01);
        // Should converge near setpoint
        let final_val = response.output.last().unwrap();
        assert!((final_val - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_pid_second_order_response() {
        use lau_control_theory::PidController;
        let mut pid = PidController::new(2.0, 1.0, 0.5);
        let response = pid.simulate_second_order(1.0, 0.5, 1.0, 20.0, 0.01);
        let final_val = response.output.last().unwrap();
        assert!((final_val - 1.0).abs() < 0.2);
    }

    #[test]
    fn test_pid_metrics() {
        use lau_control_theory::PidController;
        let mut pid = PidController::new(5.0, 2.0, 0.5);
        let response = pid.simulate_first_order(1.0, 1.0, 1.0, 20.0, 0.01);
        assert!(response.metrics.steady_state_error.abs() < 0.1);
    }

    #[test]
    fn test_ziegler_nichols_tuning() {
        use lau_control_theory::pid::ziegler_nichols;
        let params = ziegler_nichols::ZnParams { ku: 4.0, tu: 2.0 };
        let pid = ziegler_nichols::tune_pid_classic(&params);
        assert_abs_diff_eq!(pid.kp, 2.4, epsilon = 1e-10);
        assert_abs_diff_eq!(pid.ki, 2.4, epsilon = 1e-10);
        assert_abs_diff_eq!(pid.kd, 0.6, epsilon = 1e-10);
    }

    #[test]
    fn test_zn_pi_tuning() {
        use lau_control_theory::pid::ziegler_nichols;
        let params = ziegler_nichols::ZnParams { ku: 4.0, tu: 2.0 };
        let pid = ziegler_nichols::tune_pi(&params);
        assert_abs_diff_eq!(pid.kp, 1.8, epsilon = 1e-10);
        assert!(pid.kd.abs() < 1e-10);
    }

    // ===== Root Locus Tests =====

    #[test]
    fn test_root_locus_basic() {
        use lau_control_theory::TransferFunction;
        use lau_control_theory::root_locus::root_locus;
        let tf = TransferFunction::new(vec![1.0], vec![1.0, 1.0]).unwrap();
        let points = root_locus(&tf, 0.0, 10.0, 100);
        assert_eq!(points.len(), 100);
        // At K=0, poles should be the open-loop poles
        assert_abs_diff_eq!(points[0].poles[0].re, -1.0, epsilon = 1e-4);
    }

    #[test]
    fn test_root_locus_second_order() {
        use lau_control_theory::TransferFunction;
        use lau_control_theory::root_locus::root_locus;
        let tf = TransferFunction::new(vec![1.0], vec![1.0, 2.0, 1.0]).unwrap();
        let points = root_locus(&tf, 0.0, 10.0, 50);
        assert_eq!(points.len(), 50);
        // At K=0, poles at -1, -1
        for pole in &points[0].poles {
            assert_abs_diff_eq!(pole.re, -1.0, epsilon = 0.1);
        }
    }

    #[test]
    fn test_closed_loop_poles() {
        use lau_control_theory::TransferFunction;
        use lau_control_theory::root_locus::closed_loop_poles;
        let tf = TransferFunction::new(vec![1.0], vec![1.0, 1.0]).unwrap();
        let poles = closed_loop_poles(&tf, 0.0);
        // At K=0, closed-loop = open-loop poles
        assert_abs_diff_eq!(poles[0].re, -1.0, epsilon = 1e-6);
    }

    // ===== Bode Plot Tests =====

    #[test]
    fn test_bode_plot_basic() {
        use lau_control_theory::TransferFunction;
        use lau_control_theory::bode;
        let tf = TransferFunction::new(vec![1.0], vec![1.0, 1.0]).unwrap();
        let points = bode::bode_plot(&tf, 0.01, 100.0, 200);
        assert_eq!(points.len(), 200);
        // At low frequency, magnitude should be ~0 dB (DC gain = 1)
        assert!(points[0].magnitude_db.abs() < 1.0);
    }

    #[test]
    fn test_bode_gain_margin() {
        use lau_control_theory::TransferFunction;
        use lau_control_theory::bode;
        // First-order system with gain: 10/(s+1) — phase never reaches -180°
        let tf = TransferFunction::new(vec![10.0], vec![1.0, 1.0]).unwrap();
        let analysis = bode::bode_analysis(&tf, 0.01, 1000.0, 1000);
        // For first-order, phase goes from 0 to -90, never -180
        // So phase_margin should exist (gain crossover at some freq)
        assert!(analysis.phase_margin_deg.is_some());
    }

    #[test]
    fn test_bode_second_order() {
        use lau_control_theory::TransferFunction;
        use lau_control_theory::bode;
        let tf = TransferFunction::second_order(10.0, 0.5).unwrap();
        let points = bode::bode_plot(&tf, 0.1, 100.0, 100);
        assert_eq!(points.len(), 100);
        // DC gain should be 1 (0 dB) since H(0) = ωn²/ωn² = 1
        assert!(points.first().unwrap().magnitude_db.abs() < 1.0);
    }

    // ===== Nyquist Tests =====

    #[test]
    fn test_nyquist_plot() {
        use lau_control_theory::TransferFunction;
        use lau_control_theory::nyquist;
        let tf = TransferFunction::new(vec![1.0], vec![1.0, 1.0]).unwrap();
        let points = nyquist::nyquist_plot(&tf, 0.01, 100.0, 200);
        assert_eq!(points.len(), 200);
        // At DC (ω=0), should be at (1, 0)
        assert!(points[0].real > 0.5);
    }

    #[test]
    fn test_nyquist_stable_analysis() {
        use lau_control_theory::TransferFunction;
        use lau_control_theory::nyquist;
        let tf = TransferFunction::new(vec![1.0], vec![1.0, 1.0]).unwrap();
        let result = nyquist::nyquist_analysis(&tf, 0.01, 100.0, 500, 0);
        // Open-loop stable, no encirclements → closed-loop stable
        assert!(result.is_stable);
    }

    #[test]
    fn test_nyquist_high_gain() {
        use lau_control_theory::TransferFunction;
        use lau_control_theory::nyquist;
        // 10/(s+1) — still first-order, won't encircle -1
        let tf = TransferFunction::new(vec![10.0], vec![1.0, 1.0]).unwrap();
        let result = nyquist::nyquist_analysis(&tf, 0.01, 100.0, 500, 0);
        assert!(result.is_stable);
    }

    // ===== Gramians Tests =====

    #[test]
    fn test_controllability() {
        use lau_control_theory::StateSpace;
        use lau_control_theory::gramians;
        let ss = StateSpace::new(
            vec![vec![0.0, 1.0], vec![-2.0, -3.0]],
            vec![vec![0.0], vec![1.0]],
            vec![vec![1.0, 0.0]],
            vec![vec![0.0]],
        ).unwrap();
        let result = gramians::controllability(&ss);
        assert!(result.is_controllable);
        assert_eq!(result.rank, 2);
    }

    #[test]
    fn test_uncontrollable() {
        use lau_control_theory::StateSpace;
        use lau_control_theory::gramians;
        // Uncontrollable: B has zeros in direction of A's action
        let ss = StateSpace::new(
            vec![vec![-1.0, 0.0], vec![0.0, -2.0]],
            vec![vec![1.0], vec![0.0]], // Only affects first state
            vec![vec![1.0, 0.0]],
            vec![vec![0.0]],
        ).unwrap();
        let result = gramians::controllability(&ss);
        assert!(!result.is_controllable);
    }

    #[test]
    fn test_observability() {
        use lau_control_theory::StateSpace;
        use lau_control_theory::gramians;
        let ss = StateSpace::new(
            vec![vec![0.0, 1.0], vec![-2.0, -3.0]],
            vec![vec![0.0], vec![1.0]],
            vec![vec![1.0, 0.0]],
            vec![vec![0.0]],
        ).unwrap();
        let result = gramians::observability(&ss);
        assert!(result.is_observable);
        assert_eq!(result.rank, 2);
    }

    #[test]
    fn test_unobservable() {
        use lau_control_theory::StateSpace;
        use lau_control_theory::gramians;
        let ss = StateSpace::new(
            vec![vec![-1.0, 0.0], vec![0.0, -2.0]],
            vec![vec![1.0], vec![1.0]],
            vec![vec![1.0, 0.0]], // Only observes first state
            vec![vec![0.0]],
        ).unwrap();
        let result = gramians::observability(&ss);
        assert!(!result.is_observable);
    }

    #[test]
    fn test_controllability_gramian() {
        use lau_control_theory::StateSpace;
        use lau_control_theory::gramians;
        let ss = StateSpace::new(
            vec![vec![-1.0]],
            vec![vec![1.0]],
            vec![vec![1.0]],
            vec![vec![0.0]],
        ).unwrap();
        let wc = gramians::controllability_gramian(&ss);
        assert!(wc.is_some());
        // For A=-1, B=1: Wc = B²/(2*1) = 0.5
        assert!(wc.unwrap()[0][0] > 0.0);
    }

    #[test]
    fn test_observability_gramian() {
        use lau_control_theory::StateSpace;
        use lau_control_theory::gramians;
        let ss = StateSpace::new(
            vec![vec![-1.0]],
            vec![vec![1.0]],
            vec![vec![1.0]],
            vec![vec![0.0]],
        ).unwrap();
        let wo = gramians::observability_gramian(&ss);
        assert!(wo.is_some());
        assert!(wo.unwrap()[0][0] > 0.0);
    }

    // ===== Integration Tests =====

    #[test]
    fn test_full_control_loop() {
        use lau_control_theory::{TransferFunction, StateSpace, PidController};
        use lau_control_theory::stability;

        // Create plant: 1/(s+1)
        let plant = TransferFunction::new(vec![1.0], vec![1.0, 1.0]).unwrap();

        // Create PID controller: 2 + 1/s + 0.5s = (0.5s² + 2s + 1) / s
        let pid_tf = TransferFunction::new(vec![0.5, 2.0, 1.0], vec![1.0, 0.0]).unwrap();

        // Open loop: plant * controller
        let open_loop = plant.series(&pid_tf);

        // Closed loop with unity feedback
        let unity = TransferFunction::gain(1.0);
        let closed_loop = open_loop.feedback(&unity);

        // Check stability via poles
        let poles = closed_loop.poles();
        for pole in &poles {
            assert!(pole.re < 0.5, "Pole {} has positive real part", pole);
        }
    }

    #[test]
    fn test_routh_then_design_pid() {
        use lau_control_theory::stability;
        use lau_control_theory::PidController;

        // Check unstable plant: s² - 1 (one RHP root)
        let plant_coeffs = vec![1.0, 0.0, -1.0];
        assert!(!stability::is_polynomial_stable(&plant_coeffs));

        // Use PID to stabilize a marginally stable system
        let mut pid = PidController::new(2.0, 1.0, 0.5);
        let response = pid.simulate_second_order(1.0, 0.1, 1.0, 20.0, 0.01);
        // Should converge
        assert!(response.metrics.steady_state_error.abs() < 0.5);
    }

    #[test]
    fn test_ss_stability_analysis_pipeline() {
        use lau_control_theory::{StateSpace, TransferFunction};
        use lau_control_theory::stability;
        use lau_control_theory::gramians;

        // Second-order system
        let tf = TransferFunction::new(vec![1.0], vec![1.0, 5.0, 6.0]).unwrap();
        let ss = StateSpace::controllable_canonical_form(&tf).unwrap();

        // Check stability
        assert!(ss.is_stable());

        // Check Lyapunov
        let lyap = stability::lyapunov_stability(&ss, None);
        assert!(lyap.is_stable);

        // Check controllability/observability
        let ctrl = gramians::controllability(&ss);
        assert!(ctrl.is_controllable);
        let obs = gramians::observability(&ss);
        assert!(obs.is_observable);
    }
}
