//! # lau-control-theory
//!
//! Classical and modern control theory — stability, feedback, and robustness.
//!
//! Implements transfer functions, state-space representations, stability analysis,
//! PID controller design, root locus, Bode/Nyquist plots, and agent feedback control.

pub mod transfer_function;
pub mod state_space;
pub mod stability;
pub mod pid;
pub mod root_locus;
pub mod bode;
pub mod nyquist;
pub mod gramians;
pub mod agent_control;

pub use transfer_function::TransferFunction;
pub use state_space::StateSpace;
pub use pid::PidController;
