# lau-control-theory

> Classical and modern control theory — stability, feedback, and robustness in Rust

## What This Does

Classical and modern control theory — stability, feedback, and robustness in Rust. Part of the PLATO/LAU ecosystem — a mathematically rigorous framework for building educational agents that learn, teach, and evolve.

## The Key Idea

This crate implements the core abstractions needed for its domain, with a focus on correctness, composability, and conservation guarantees. Every public type is serializable (serde), every algorithm is tested, and every invariant is verified.

## Install

```bash
cargo add lau-control-theory
```

## Quick Start

See the API Reference below for complete usage. Key entry points:

```rust
use lau_control_theory::*;
// See types and methods below for complete usage
```

## API Reference

```rust
pub struct RouthHurwitzResult 
pub fn routh_hurwitz(coeffs: &[f64]) -> Result<RouthHurwitzResult, String> 
pub fn is_polynomial_stable(coeffs: &[f64]) -> bool 
pub fn hurwitz_matrix(coeffs: &[f64]) -> Vec<Vec<f64>> 
pub struct LyapunovResult 
pub fn lyapunov_stability(ss: &StateSpace, q: Option<&[Vec<f64>]>) -> LyapunovResult 
pub struct ControllabilityResult 
pub struct ObservabilityResult 
pub fn controllability(ss: &StateSpace) -> ControllabilityResult 
pub fn observability(ss: &StateSpace) -> ObservabilityResult 
pub fn controllability_gramian(ss: &StateSpace) -> Option<Vec<Vec<f64>>> 
pub fn observability_gramian(ss: &StateSpace) -> Option<Vec<Vec<f64>>> 
pub struct TransferFunction 
    pub fn new(num: Vec<f64>, den: Vec<f64>) -> Result<Self, String> 
    pub fn gain(k: f64) -> Self 
    pub fn first_order(k: f64, tau: f64) -> Result<Self, String> 
    pub fn second_order(wn: f64, zeta: f64) -> Result<Self, String> 
    pub fn evaluate(&self, s: Complex64) -> Complex64 
    pub fn dc_gain(&self) -> f64 
    pub fn poles(&self) -> Vec<Complex64> 
    pub fn zeros(&self) -> Vec<Complex64> 
    pub fn frequency_response(&self, omega: f64) -> Complex64 
    pub fn series(&self, other: &TransferFunction) -> TransferFunction 
    pub fn parallel(&self, other: &TransferFunction) -> TransferFunction 
    pub fn feedback(&self, controller: &TransferFunction) -> TransferFunction 
    pub fn order(&self) -> usize 
    pub fn normalize(&mut self) 
pub struct BodePoint 
pub struct BodeAnalysis 
pub fn bode_plot(tf: &TransferFunction, omega_min: f64, omega_max: f64, n_points: usize) -> Vec<BodePoint> 
pub fn bode_analysis(tf: &TransferFunction, omega_min: f64, omega_max: f64, n_points: usize) -> BodeAnalysis 
pub struct NyquistPoint 
pub struct NyquistResult 
pub fn nyquist_plot(tf: &TransferFunction, omega_min: f64, omega_max: f64, n_points: usize) -> Vec<NyquistPoint> 
pub fn nyquist_analysis(
pub fn passes_through_critical(points: &[NyquistPoint], tolerance: f64) -> bool 
pub struct PidController 
    pub fn new(kp: f64, ki: f64, kd: f64) -> Self 
    pub fn p_only(kp: f64) -> Self 
    pub fn pi(kp: f64, ki: f64) -> Self 
    pub fn pd(kp: f64, kd: f64) -> Self 
    pub fn update(&mut self, error: f64, dt: f64) -> f64 
    pub fn integral_value(&self) -> f64 
    pub fn reset(&mut self) 
    pub fn simulate_first_order(
    pub fn simulate_second_order(
    pub fn compute_metrics(
pub struct PidResponse 
pub struct PidMetrics 
    pub struct ZnParams 
    pub fn tune_pid_classic(params: &ZnParams) -> PidController 
    pub fn tune_pi(params: &ZnParams) -> PidController 
    pub fn tune_p(params: &ZnParams) -> PidController 
    pub fn tune_pid_no_overshoot(params: &ZnParams) -> PidController 
    pub fn tune_pid_some_overshoot(params: &ZnParams) -> PidController 
pub struct RootLocusPoint 
pub fn root_locus(
pub fn closed_loop_poles(tf: &TransferFunction, k: f64) -> Vec<Complex64> 
pub fn critical_gain(tf: &TransferFunction) -> Option<f64> 
pub fn imaginary_axis_crossings(tf: &TransferFunction, n_points: usize) -> Vec<(f64, Complex64)> 
```

## How It Works

Read the source in `src/` for full implementation details. All algorithms are documented with inline comments explaining the mathematical foundations.

## The Math

This crate implements formal mathematical constructs. See the source documentation for theorem statements and proofs of correctness.

## Testing

**65 tests** covering construction, serialization, correctness properties, edge cases, and composability with other lau-* crates.

## License

MIT
