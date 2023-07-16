mod anneal;

use std::time::{Duration, Instant};

use anneal::run_annealing;
use rand::{
    rngs::{SmallRng, StdRng},
    RngCore, SeedableRng,
};

/// Holds extra information given to [`AnnealingStrategy`].
pub struct AnnealingContext {
    /// Energy of the best state the current thread has seen so far.
    /// It is energy of the best state, not necessarily the best solution, so
    /// it may represent an invalid solution.
    pub best_energy: f64,
    pub environment: Environment,
    pub thread_id: usize,
}

/// Represents a state of simulated annealing.
pub trait State: Send {
    type Solution: Send;

    /// Computes the energy of the state. This method must return very quickly,
    /// preferably just by returning a cached value.
    fn energy(&self) -> f64;

    /// Creates a solution from the state. It should return [`None`] if the
    /// state corresponds to an invalid solution.
    fn as_solution(&self) -> Option<Self::Solution>;
}

/// Represents a neighbour of an annealing state.
///
/// It is similar to [`State`] as both should be able to compute energy, but
/// `StateNeighbour` is meant to converted back to [`State`] immediately either
/// by `accept` or `revert`.
pub trait StateNeighbour<S> {
    /// Computes the energy of the state. This method must return very quickly,
    /// preferably just by returning a cached value.
    fn energy(&self) -> f64;

    /// Accepts this neighbour and returns the corresponding state.
    fn accept(self) -> S;

    /// Rejects this neighbour and returns the original state.
    fn revert(self) -> S;
}

/// Defines neighbors of an annealing state and how to transition among them.
pub trait AnnealingStrategy<S, R = SmallRng>: Sync
where
    S: State,
{
    type Neighbour: StateNeighbour<S>;

    /// Randomly chooses a neighbour of the given state.
    fn neighbour(&self, state: S, context: &AnnealingContext, rng: &mut R) -> Self::Neighbour;
}

/// Represents the status of simulated annealing at a point of time.
#[derive(Copy, Clone, Debug)]
pub struct Environment {
    /// A floating-point value in the range of `[0, 1)` indicating the progress
    /// of the simulated annealing process.
    pub progress_ratio: f64,

    /// The current temperature of the simulated annealing process.
    pub temperature: f64,
}

/// Computes [`Environment`]. Also known as a cooling strategy.
pub trait EnvironmentStrategy {
    /// Computes the current [`Environment`]. It should return [`None`] when the
    /// simulated annealing process has finished.
    fn environment(&self) -> Option<Environment>;
}

/// An implementation of [`EnvironmentStrategy`] that cools down the temperature
/// exponentially for a fixed time limit.
pub struct ExponentialCoolingStrategy {
    initial_temperature: f64,
    final_temperature: f64,
    start_time: Instant,
    time_limit: Duration,
}

impl ExponentialCoolingStrategy {
    pub fn new(initial_temperature: f64, final_temperature: f64, time_limit: Duration) -> Self {
        let start_time = Instant::now();
        Self {
            initial_temperature,
            final_temperature,
            start_time,
            time_limit,
        }
    }
}

impl EnvironmentStrategy for ExponentialCoolingStrategy {
    fn environment(&self) -> Option<Environment> {
        let progress_ratio =
            self.start_time.elapsed().as_secs_f64() / self.time_limit.as_secs_f64();
        if progress_ratio >= 1.0 {
            return None;
        }

        let temperature = self.initial_temperature
            * (self.final_temperature / self.initial_temperature).powf(progress_ratio);
        Some(Environment {
            progress_ratio,
            temperature,
        })
    }
}

/// An implementation of [`EnvironmentStrategy`] that cools down the temperature
/// linearly for a fixed time limit.
pub struct LinearCoolingStrategy {
    initial_temperature: f64,
    final_temperature: f64,
    start_time: Instant,
    time_limit: Duration,
}

impl LinearCoolingStrategy {
    pub fn new(initial_temperature: f64, final_temperature: f64, time_limit: Duration) -> Self {
        let start_time = Instant::now();
        Self {
            initial_temperature,
            final_temperature,
            start_time,
            time_limit,
        }
    }
}

impl EnvironmentStrategy for LinearCoolingStrategy {
    fn environment(&self) -> Option<Environment> {
        let progress_ratio =
            self.start_time.elapsed().as_secs_f64() / self.time_limit.as_secs_f64();
        if progress_ratio >= 1.0 {
            return None;
        }

        let temperature = self.initial_temperature * (1.0 - progress_ratio)
            + self.final_temperature * progress_ratio;
        Some(Environment {
            progress_ratio,
            temperature,
        })
    }
}

/// Responsible for generating initial [`State`]s for simulated annealing.
///
/// A single `StateInitializer` is shared to generate initial states for all
/// threads, so it must implement [`Sync`].
pub trait StateInitializer<S, R = StdRng>: Sync {
    /// Generates an initial state for a simulated annealing thread.
    fn generate_initial_state(&self, rng: &mut R) -> S;
}

/// Monitors the progress of annealing threads.
///
/// On every iteration, every annealing thread calls `ProgressMonitor` to notify
/// it the current state. `ProgressMonitor` may report to users. Note that an
/// annealing thread is blocked until `ProgressMonitor` returns, so its
/// implementations should try to return very quickly in most calls, e.g. by
/// only inspecting the given state on every N calls.
///
/// A single `ProgressMonitor` is shared among all annealing threads, so it must
/// implement [`Sync`].
pub trait ProgressMonitor<S>: Sync {
    fn progress(&self, state: &S, context: &AnnealingContext);
}

/// An implementation of [`ProgressMonitor`] that does nothing.
pub struct NullProgressMonitor;

impl<S> ProgressMonitor<S> for NullProgressMonitor {
    fn progress(&self, _state: &S, _context: &AnnealingContext) {}
}

/// A result of simulated annealing returned by [`anneal`].
pub struct AnnealingResult<S>
where
    S: State,
{
    pub best_solution: Option<S::Solution>,
    pub best_energy: f64,
    pub final_states: Vec<S>,
    pub total_iterations: u64,
}

/// Parameters to [`anneal`].
pub struct AnnealingParams<AS, ES, SI, PM, RR> {
    pub annealing_strategy: AS,
    pub environment_strategy: ES,
    pub state_initializer: SI,
    pub progress_monitor: PM,
    pub root_rng: RR,
    pub threads: usize,
}

/// Runs simulated annealing.
pub fn anneal<S, AS, ES, SI, PM, RR, TR>(
    params: AnnealingParams<AS, ES, SI, PM, RR>,
) -> AnnealingResult<S>
where
    S: State,
    AS: AnnealingStrategy<S, TR>,
    ES: EnvironmentStrategy,
    SI: StateInitializer<S, RR>,
    PM: ProgressMonitor<S>,
    RR: RngCore,
    TR: RngCore + SeedableRng + Send,
{
    run_annealing(params)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng as _;

    // The objective function to minimize: (x-1)^2
    fn energy(x: f64) -> f64 {
        (x - 1.0).powi(2)
    }

    #[derive(Clone)]
    struct TestState {
        x: f64,
    }

    impl State for TestState {
        type Solution = f64;

        fn energy(&self) -> f64 {
            energy(self.x)
        }

        fn as_solution(&self) -> Option<Self::Solution> {
            Some(self.x)
        }
    }

    struct TestNeighbour {
        new_x: f64,
        old_x: f64,
    }

    impl StateNeighbour<TestState> for TestNeighbour {
        fn energy(&self) -> f64 {
            energy(self.new_x)
        }

        fn accept(self) -> TestState {
            TestState { x: self.new_x }
        }

        fn revert(self) -> TestState {
            TestState { x: self.old_x }
        }
    }

    struct TestStrategy;

    impl AnnealingStrategy<TestState> for TestStrategy {
        type Neighbour = TestNeighbour;

        fn neighbour(
            &self,
            state: TestState,
            _context: &AnnealingContext,
            rng: &mut SmallRng,
        ) -> TestNeighbour {
            let dx = rng.gen_range(-1.0..=1.0);
            TestNeighbour {
                new_x: state.x + dx,
                old_x: state.x,
            }
        }
    }

    struct TestStateInitializer;

    impl StateInitializer<TestState> for TestStateInitializer {
        fn generate_initial_state(&self, rng: &mut StdRng) -> TestState {
            let x = rng.gen::<f64>();
            TestState { x }
        }
    }

    #[test]
    fn test_anneal() {
        let result = anneal(AnnealingParams {
            annealing_strategy: TestStrategy,
            environment_strategy: ExponentialCoolingStrategy::new(
                1.0,
                0.0000001,
                Duration::from_secs(1),
            ),
            state_initializer: TestStateInitializer,
            progress_monitor: NullProgressMonitor,
            root_rng: StdRng::from_seed(Default::default()),
            threads: 2,
        });

        let x = result.best_solution.unwrap();

        // Technically this is a flaky test, but it should reliably pass since
        // the objective function is very simple.
        assert!(
            (0.9..=1.1).contains(&x),
            "Annealing result = {:.3}; want ~1.0 ({} iterations)",
            x,
            result.total_iterations
        );
    }
}
