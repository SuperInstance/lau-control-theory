//! Agent feedback control — keeping agent behavior within desired bounds.
//!
//! This module applies control theory concepts to agent behavior management,
//! treating an agent as a dynamical system that needs to be kept within
//! safe operating bounds.

use serde::{Deserialize, Serialize};
use crate::pid::PidController;

/// Metrics describing an agent's behavioral state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentMetrics {
    /// Token usage rate (tokens per minute)
    pub token_rate: f64,
    /// Error rate (errors per interaction)
    pub error_rate: f64,
    /// Response latency in seconds
    pub latency: f64,
    /// Deviation score from expected behavior (0.0 = perfect)
    pub deviation_score: f64,
}

impl Default for AgentMetrics {
    fn default() -> Self {
        Self {
            token_rate: 0.0,
            error_rate: 0.0,
            latency: 0.0,
            deviation_score: 0.0,
        }
    }
}

/// Bounds for acceptable agent behavior.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BehaviorBounds {
    /// Maximum allowed token rate
    pub max_token_rate: f64,
    /// Maximum allowed error rate
    pub max_error_rate: f64,
    /// Maximum allowed latency
    pub max_latency: f64,
    /// Maximum allowed deviation score
    pub max_deviation: f64,
}

impl Default for BehaviorBounds {
    fn default() -> Self {
        Self {
            max_token_rate: 1000.0,
            max_error_rate: 0.1,
            max_latency: 30.0,
            max_deviation: 0.5,
        }
    }
}

/// Control action to apply to an agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ControlAction {
    /// Throttle factor (0.0 = full stop, 1.0 = no throttle)
    pub throttle: f64,
    /// Whether to apply stricter prompting
    pub strict_mode: bool,
    /// Warning level (0 = none, 1 = mild, 2 = severe)
    pub warning_level: u32,
    /// Description of the control action
    pub description: String,
}

impl Default for ControlAction {
    fn default() -> Self {
        Self {
            throttle: 1.0,
            strict_mode: false,
            warning_level: 0,
            description: "Normal operation".into(),
        }
    }
}

/// Agent feedback controller using PID control.
pub struct AgentFeedbackController {
    /// PID controller for token rate
    token_pid: PidController,
    /// PID controller for error rate
    error_pid: PidController,
    /// PID controller for deviation
    deviation_pid: PidController,
    /// Behavior bounds
    bounds: BehaviorBounds,
    /// Time step in seconds
    dt: f64,
}

impl AgentFeedbackController {
    /// Create a new agent feedback controller with the given bounds.
    pub fn new(bounds: BehaviorBounds, dt: f64) -> Self {
        // PID gains tuned for each dimension
        let token_pid = PidController::new(0.5, 0.1, 0.05);
        let error_pid = PidController::new(2.0, 0.5, 0.1);
        let deviation_pid = PidController::new(1.5, 0.3, 0.2);

        Self {
            token_pid,
            error_pid,
            deviation_pid,
            bounds,
            dt,
        }
    }

    /// Create with default bounds and PID parameters.
    pub fn with_defaults() -> Self {
        Self::new(BehaviorBounds::default(), 1.0)
    }

    /// Compute a control action based on current agent metrics.
    ///
    /// Returns a ControlAction that describes how to adjust the agent's behavior.
    pub fn control(&mut self, metrics: &AgentMetrics) -> ControlAction {
        // Compute errors (how far above the bound we are, 0 if within bounds)
        let token_error = (metrics.token_rate - self.bounds.max_token_rate).max(0.0);
        let error_error = (metrics.error_rate - self.bounds.max_error_rate).max(0.0);
        let deviation_error = (metrics.deviation_score - self.bounds.max_deviation).max(0.0);

        // Get PID outputs (negative = reduce, positive = increase)
        let token_correction = self.token_pid.update(token_error, self.dt);
        let error_correction = self.error_pid.update(error_error, self.dt);
        let deviation_correction = self.deviation_pid.update(deviation_error, self.dt);

        // Compute overall throttle from PID corrections
        // Positive corrections mean we need to reduce throttle
        let total_correction = token_correction + error_correction + deviation_correction;
        let throttle = (1.0 - total_correction * 0.1).clamp(0.0, 1.0);

        // Determine warning level
        let violations = [
            metrics.token_rate > self.bounds.max_token_rate,
            metrics.error_rate > self.bounds.max_error_rate,
            metrics.deviation_score > self.bounds.max_deviation,
        ].iter().filter(|&&v| v).count();

        let (warning_level, description) = match violations {
            0 => (0, "Normal operation".into()),
            1 => (1, "Mild: one bound exceeded".into()),
            2 => (2, "Severe: two bounds exceeded".into()),
            _ => (2, "Critical: all bounds exceeded".into()),
        };

        ControlAction {
            throttle,
            strict_mode: violations >= 2,
            warning_level,
            description,
        }
    }

    /// Check if metrics are within bounds (no control needed).
    pub fn is_within_bounds(&self, metrics: &AgentMetrics) -> bool {
        metrics.token_rate <= self.bounds.max_token_rate
            && metrics.error_rate <= self.bounds.max_error_rate
            && metrics.latency <= self.bounds.max_latency
            && metrics.deviation_score <= self.bounds.max_deviation
    }

    /// Reset all PID controllers.
    pub fn reset(&mut self) {
        self.token_pid.reset();
        self.error_pid.reset();
        self.deviation_pid.reset();
    }
}

/// Stability analysis for agent behavior over time.
pub fn analyze_agent_stability(
    metric_history: &[AgentMetrics],
    bounds: &BehaviorBounds,
) -> AgentStabilityReport {
    if metric_history.is_empty() {
        return AgentStabilityReport {
            is_stable: true,
            bound_violations: 0,
            oscillation_detected: false,
            mean_deviation: 0.0,
        };
    }

    let n = metric_history.len();
    let mut violations = 0;
    let mut deviations = Vec::new();

    for m in metric_history {
        let mut dev = 0.0;
        if m.token_rate > bounds.max_token_rate {
            violations += 1;
            dev += (m.token_rate - bounds.max_token_rate) / bounds.max_token_rate;
        }
        if m.error_rate > bounds.max_error_rate {
            violations += 1;
            dev += (m.error_rate - bounds.max_error_rate) / bounds.max_error_rate;
        }
        if m.deviation_score > bounds.max_deviation {
            violations += 1;
            dev += (m.deviation_score - bounds.max_deviation) / bounds.max_deviation;
        }
        deviations.push(dev);
    }

    let mean_deviation = deviations.iter().sum::<f64>() / n as f64;

    // Detect oscillation: check if the signal changes direction frequently
    let oscillation_detected = detect_oscillation(&deviations);

    let is_stable = violations == 0 && !oscillation_detected;

    AgentStabilityReport {
        is_stable,
        bound_violations: violations,
        oscillation_detected,
        mean_deviation,
    }
}

/// Report on agent behavioral stability.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentStabilityReport {
    /// Whether the agent's behavior is stable
    pub is_stable: bool,
    /// Total number of bound violations
    pub bound_violations: usize,
    /// Whether oscillatory behavior was detected
    pub oscillation_detected: bool,
    /// Mean deviation from bounds (normalized)
    pub mean_deviation: f64,
}

/// Detect oscillation in a signal by counting zero crossings of the detrended signal.
fn detect_oscillation(signal: &[f64]) -> bool {
    if signal.len() < 4 {
        return false;
    }

    let mean = signal.iter().sum::<f64>() / signal.len() as f64;
    let detrended: Vec<f64> = signal.iter().map(|&x| x - mean).collect();

    let mut crossings = 0;
    for i in 1..detrended.len() {
        if detrended[i - 1] * detrended[i] < 0.0 {
            crossings += 1;
        }
    }

    // If more than half the points are sign changes, it's oscillatory
    crossings > signal.len() / 2
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agent_within_bounds() {
        let controller = AgentFeedbackController::with_defaults();
        let metrics = AgentMetrics {
            token_rate: 500.0,
            error_rate: 0.01,
            latency: 5.0,
            deviation_score: 0.1,
        };
        assert!(controller.is_within_bounds(&metrics));
    }

    #[test]
    fn test_agent_outside_bounds() {
        let controller = AgentFeedbackController::with_defaults();
        let metrics = AgentMetrics {
            token_rate: 2000.0,
            error_rate: 0.5,
            latency: 60.0,
            deviation_score: 0.9,
        };
        assert!(!controller.is_within_bounds(&metrics));
    }

    #[test]
    fn test_feedback_control_normal() {
        let mut controller = AgentFeedbackController::with_defaults();
        let metrics = AgentMetrics {
            token_rate: 500.0,
            error_rate: 0.01,
            latency: 5.0,
            deviation_score: 0.1,
        };
        let action = controller.control(&metrics);
        assert!(!action.strict_mode);
        assert_eq!(action.warning_level, 0);
    }

    #[test]
    fn test_feedback_control_throttle() {
        let mut controller = AgentFeedbackController::with_defaults();
        // Feed multiple high-error samples to build up PID response
        for _ in 0..10 {
            let metrics = AgentMetrics {
                token_rate: 2000.0,
                error_rate: 0.5,
                latency: 60.0,
                deviation_score: 0.9,
            };
            let action = controller.control(&metrics);
            if action.throttle < 1.0 {
                return; // success
            }
        }
        panic!("Throttle should have decreased after repeated violations");
    }

    #[test]
    fn test_agent_stability_analysis() {
        let metrics = vec![
            AgentMetrics { token_rate: 500.0, error_rate: 0.01, latency: 5.0, deviation_score: 0.1 },
            AgentMetrics { token_rate: 600.0, error_rate: 0.02, latency: 6.0, deviation_score: 0.15 },
            AgentMetrics { token_rate: 550.0, error_rate: 0.015, latency: 5.5, deviation_score: 0.12 },
        ];
        let bounds = BehaviorBounds::default();
        let report = analyze_agent_stability(&metrics, &bounds);
        assert!(report.is_stable);
        assert_eq!(report.bound_violations, 0);
    }

    #[test]
    fn test_oscillation_detection() {
        let signal = vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
        assert!(detect_oscillation(&signal));

        let stable = vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5];
        assert!(!detect_oscillation(&stable));
    }
}
