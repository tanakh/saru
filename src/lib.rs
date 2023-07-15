mod anneal;

use std::time::{Duration, Instant};

use anneal::run_annealing;
use rand::{
    rngs::{SmallRng, StdRng},
    RngCore, SeedableRng,
};

pub struct AnnealingContext {
    pub best_energy: f64,
    pub environment: Environment,
    pub thread_id: usize,
}

pub trait State: Send {
    type Solution: Send;

    fn energy(&self) -> f64;
    fn as_solution(&self) -> Option<Self::Solution>;
}

pub trait StateNeighbor<S> {
    fn energy(&self) -> f64;
    fn accept(self) -> S;
    fn revert(self) -> S;
}

pub trait AnnealingStrategy<S, R = SmallRng>: Sync
where
    S: State,
{
    type Neighbour: StateNeighbor<S>;

    fn neighbour(&self, state: S, context: &AnnealingContext, rng: &mut R) -> Self::Neighbour;
}

#[derive(Copy, Clone, Debug)]
pub struct Environment {
    pub progress_ratio: f64,
    pub temperature: f64,
}

pub trait EnvironmentStrategy {
    fn environment(&self) -> Option<Environment>;
}

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

pub trait StateInitializer<S, R = StdRng>: Sync {
    fn generate_initial_state(&self, rng: &mut R) -> S;
}

pub trait ProgressMonitor<S>: Sync {
    fn progress(&self, state: &S, context: &AnnealingContext);
}

pub struct NullProgressMonitor;

impl<S> ProgressMonitor<S> for NullProgressMonitor {
    fn progress(&self, _state: &S, _context: &AnnealingContext) {}
}

pub struct AnnealingResult<S>
where
    S: State,
{
    pub best_solution: Option<S::Solution>,
    pub best_energy: f64,
    pub final_states: Vec<S>,
    pub total_iterations: u64,
}

pub struct AnnealingParams<AS, ES, SI, PM, RR> {
    pub annealing_strategy: AS,
    pub environment_strategy: ES,
    pub state_initializer: SI,
    pub progress_monitor: PM,
    pub root_rng: RR,
    pub threads: usize,
}

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

    struct TestNeighbor {
        new_x: f64,
        old_x: f64,
    }

    impl StateNeighbor<TestState> for TestNeighbor {
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
        type Neighbour = TestNeighbor;

        fn neighbour(
            &self,
            state: TestState,
            _context: &AnnealingContext,
            rng: &mut SmallRng,
        ) -> TestNeighbor {
            let dx = rng.gen_range(-1.0..=1.0);
            TestNeighbor {
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
