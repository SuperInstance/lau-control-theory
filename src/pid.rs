//! PID controller design, tuning, and response analysis.

use serde::{Deserialize, Serialize};

/// PID controller parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PidController {
    /// Proportional gain
    pub kp: f64,
    /// Integral gain
    pub ki: f64,
    /// Derivative gain
    pub kd: f64,
    /// Output limits (min, max). None = no limits.
    pub output_limits: Option<(f64, f64)>,
    /// Anti-windup: integral term limit
    pub integral_limit: Option<f64>,
    /// Derivative filter coefficient (for filtered derivative)
    pub derivative_filter: Option<f64>,

    // Internal state
    #[serde(skip)]
    integral: f64,
    #[serde(skip)]
    prev_error: f64,
    #[serde(skip)]
    prev_derivative: f64,
    #[serde(skip)]
    initialized: bool,
}

impl PidController {
    /// Create a new PID controller with the given gains.
    pub fn new(kp: f64, ki: f64, kd: f64) -> Self {
        Self {
            kp,
            ki,
            kd,
            output_limits: None,
            integral_limit: None,
            derivative_filter: None,
            integral: 0.0,
            prev_error: 0.0,
            prev_derivative: 0.0,
            initialized: false,
        }
    }

    /// Create a proportional-only controller.
    pub fn p_only(kp: f64) -> Self {
        Self::new(kp, 0.0, 0.0)
    }

    /// Create a PI controller.
    pub fn pi(kp: f64, ki: f64) -> Self {
        Self::new(kp, ki, 0.0)
    }

    /// Create a PD controller.
    pub fn pd(kp: f64, kd: f64) -> Self {
        Self::new(kp, 0.0, kd)
    }

    /// Compute the PID output for a given error and time step.
    pub fn update(&mut self, error: f64, dt: f64) -> f64 {
        if dt <= 0.0 {
            return 0.0;
        }

        // Proportional term
        let p_term = self.kp * error;

        // Integral term with anti-windup
        self.integral += error * dt;
        if let Some(limit) = self.integral_limit {
            self.integral = self.integral.clamp(-limit, limit);
        }
        let i_term = self.ki * self.integral;

        // Derivative term (filtered)
        let raw_derivative = if self.initialized {
            (error - self.prev_error) / dt
        } else {
            0.0
        };

        let d_term = if let Some(alpha) = self.derivative_filter {
            let filtered = alpha * raw_derivative + (1.0 - alpha) * self.prev_derivative;
            self.prev_derivative = filtered;
            self.kd * filtered
        } else {
            self.kd * raw_derivative
        };

        self.prev_error = error;
        self.initialized = true;

        let output = p_term + i_term + d_term;

        // Apply output limits
        if let Some((min, max)) = self.output_limits {
            output.clamp(min, max)
        } else {
            output
        }
    }

    /// Get the current integral accumulator value.
    pub fn integral_value(&self) -> f64 {
        self.integral
    }

    /// Reset the controller state.
    pub fn reset(&mut self) {
        self.integral = 0.0;
        self.prev_error = 0.0;
        self.prev_derivative = 0.0;
        self.initialized = false;
    }

    /// Simulate PID control of a first-order plant: G(s) = K / (τs + 1)
    pub fn simulate_first_order(
        &mut self,
        plant_gain: f64,
        plant_tau: f64,
        setpoint: f64,
        t_end: f64,
        dt: f64,
    ) -> PidResponse {
        let mut time = vec![];
        let mut output = vec![];
        let mut control_signal = vec![];

        let mut y = 0.0; // plant output
        self.reset();

        let n_steps = (t_end / dt).ceil() as usize;
        for step in 0..=n_steps {
            let t = step as f64 * dt;
            let error = setpoint - y;

            let u = self.update(error, dt);

            time.push(t);
            output.push(y);
            control_signal.push(u);

            // First-order plant: dy/dt = (plant_gain * u - y) / plant_tau
            let dy_dt = (plant_gain * u - y) / plant_tau;
            y += dy_dt * dt;
        }

        // Compute performance metrics
        let metrics = Self::compute_metrics(&time, &output, setpoint, dt);

        PidResponse {
            time,
            output,
            control_signal,
            metrics,
        }
    }

    /// Simulate PID control of a second-order plant: G(s) = ωn² / (s² + 2ζωn·s + ωn²)
    pub fn simulate_second_order(
        &mut self,
        wn: f64,
        zeta: f64,
        setpoint: f64,
        t_end: f64,
        dt: f64,
    ) -> PidResponse {
        let mut time = vec![];
        let mut output = vec![];
        let mut control_signal = vec![];

        let mut x1 = 0.0; // position
        let mut x2 = 0.0; // velocity
        self.reset();

        let wn2 = wn * wn;
        let n_steps = (t_end / dt).ceil() as usize;
        for step in 0..=n_steps {
            let t = step as f64 * dt;
            let error = setpoint - x1;

            let u = self.update(error, dt);

            time.push(t);
            output.push(x1);
            control_signal.push(u);

            // Second-order: x1' = x2, x2' = wn² * u - 2ζωn * x2 - wn² * x1
            let dx1 = x2;
            let dx2 = wn2 * u - 2.0 * zeta * wn * x2 - wn2 * x1;
            x1 += dx1 * dt;
            x2 += dx2 * dt;
        }

        let metrics = Self::compute_metrics(&time, &output, setpoint, dt);

        PidResponse {
            time,
            output,
            control_signal,
            metrics,
        }
    }

    /// Compute performance metrics from a step response.
    pub fn compute_metrics(
        time: &[f64],
        output: &[f64],
        setpoint: f64,
        _dt: f64,
    ) -> PidMetrics {
        let final_val = output.last().copied().unwrap_or(0.0);
        let steady_state_error = setpoint - final_val;

        // Rise time: time to go from 10% to 90% of setpoint
        let target_10 = setpoint * 0.1;
        let target_90 = setpoint * 0.9;
        let mut rise_time = None;
        let mut t10 = None;
        let mut t90 = None;
        for (i, &y) in output.iter().enumerate() {
            if t10.is_none() && y >= target_10 {
                t10 = Some(time[i]);
            }
            if t10.is_some() && t90.is_none() && y >= target_90 {
                t90 = Some(time[i]);
                break;
            }
        }
        if let (Some(a), Some(b)) = (t10, t90) {
            rise_time = Some(b - a);
        }

        // Settling time: time after which output stays within 2% of setpoint
        let tolerance = setpoint.abs() * 0.02;
        let mut settling_time = None;
        for i in (0..output.len()).rev() {
            if (output[i] - setpoint).abs() > tolerance {
                if i + 1 < time.len() {
                    settling_time = Some(time[i + 1]);
                }
                break;
            }
        }

        // Overshoot
        let max_val = output.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let overshoot = if setpoint > 0.0 && max_val > setpoint {
            ((max_val - setpoint) / setpoint) * 100.0
        } else {
            0.0
        };

        PidMetrics {
            rise_time,
            settling_time,
            overshoot_percent: overshoot,
            steady_state_error,
        }
    }
}

/// Result of a PID simulation.
#[derive(Debug, Clone)]
pub struct PidResponse {
    /// Time points
    pub time: Vec<f64>,
    /// Output values
    pub output: Vec<f64>,
    /// Control signal values
    pub control_signal: Vec<f64>,
    /// Performance metrics
    pub metrics: PidMetrics,
}

/// Performance metrics for a step response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PidMetrics {
    /// Rise time (10% to 90% of setpoint)
    pub rise_time: Option<f64>,
    /// Settling time (within 2% of setpoint)
    pub settling_time: Option<f64>,
    /// Overshoot as percentage
    pub overshoot_percent: f64,
    /// Steady-state error
    pub steady_state_error: f64,
}

/// Ziegler-Nichols tuning method.
pub mod ziegler_nichols {
    use super::PidController;

    /// Parameters discovered during Ziegler-Nichols oscillation test.
    #[derive(Debug, Clone)]
    pub struct ZnParams {
        /// Ultimate gain (gain at which system oscillates)
        pub ku: f64,
        /// Ultimate period (period of sustained oscillation)
        pub tu: f64,
    }

    /// Tune PID using Ziegler-Nichols "Classic" rules.
    pub fn tune_pid_classic(params: &ZnParams) -> PidController {
        let kp = 0.6 * params.ku;
        let ki = 1.2 * params.ku / params.tu;
        let kd = 0.075 * params.ku * params.tu;
        PidController::new(kp, ki, kd)
    }

    /// Tune PI using Ziegler-Nichols rules.
    pub fn tune_pi(params: &ZnParams) -> PidController {
        let kp = 0.45 * params.ku;
        let ki = 0.54 * params.ku / params.tu;
        PidController::new(kp, ki, 0.0)
    }

    /// Tune P-only using Ziegler-Nichols rules.
    pub fn tune_p(params: &ZnParams) -> PidController {
        let kp = 0.5 * params.ku;
        PidController::new(kp, 0.0, 0.0)
    }

    /// Tune PID using Ziegler-Nichols "No Overshoot" rules (Pessen).
    pub fn tune_pid_no_overshoot(params: &ZnParams) -> PidController {
        let kp = 0.2 * params.ku;
        let ki = 0.4 * params.ku / params.tu;
        let kd = 0.0666 * params.ku * params.tu;
        PidController::new(kp, ki, kd)
    }

    /// Tune PID using Ziegler-Nichols "Some Overshoot" rules.
    pub fn tune_pid_some_overshoot(params: &ZnParams) -> PidController {
        let kp = params.ku / 3.0;
        let ki = 2.0 * params.ku / (3.0 * params.tu);
        let kd = params.ku * params.tu / 9.0;
        PidController::new(kp, ki, kd)
    }
}
