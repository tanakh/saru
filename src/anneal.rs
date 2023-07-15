use std::{sync::mpsc, thread::sleep, time::Duration};

use rand::{Rng as _, RngCore, SeedableRng};

use crate::{
    AnnealingContext, AnnealingParams, AnnealingResult, AnnealingStrategy, Environment,
    EnvironmentStrategy, ProgressMonitor, State, StateInitializer, StateNeighbour,
};

struct ThreadAnnealingParams<'a, AS, S, M, R> {
    pub thread_id: usize,
    pub annealing_strategy: &'a AS,
    pub initial_state: S,
    pub progress_monitor: &'a M,
    pub rng: R,
    pub environment_receiver: mpsc::Receiver<Environment>,
}

struct ThreadAnnealingResult<S>
where
    S: State,
{
    pub best_solution: Option<S::Solution>,
    pub best_energy: f64,
    pub final_state: S,
    pub iterations: u64,
}

fn run_annealing_thread<AS, S, M, R>(
    params: ThreadAnnealingParams<AS, S, M, R>,
) -> ThreadAnnealingResult<S>
where
    AS: AnnealingStrategy<S, R>,
    S: State,
    M: ProgressMonitor<S>,
    R: RngCore,
{
    let mut rng = params.rng;

    let mut iterations = 0;
    let mut current_state = params.initial_state;
    let mut current_energy = current_state.energy();
    let mut best_solution = current_state.as_solution();

    let initial_environment = match params.environment_receiver.recv() {
        Ok(environment) => environment,
        Err(_) => {
            // An exceptional case where the annealing finished immediately.
            let best_energy = if best_solution.is_some() {
                current_energy
            } else {
                f64::INFINITY
            };
            return ThreadAnnealingResult {
                best_solution,
                best_energy,
                final_state: current_state,
                iterations,
            };
        }
    };

    let mut context = AnnealingContext {
        best_energy: if best_solution.is_some() {
            current_energy
        } else {
            f64::INFINITY
        },
        environment: initial_environment,
        thread_id: params.thread_id,
    };

    loop {
        match params.environment_receiver.try_recv() {
            Ok(environment) => {
                context.environment = environment;
            }
            Err(mpsc::TryRecvError::Disconnected) => {
                return ThreadAnnealingResult {
                    best_solution,
                    best_energy: context.best_energy,
                    final_state: current_state,
                    iterations,
                };
            }
            Err(mpsc::TryRecvError::Empty) => {}
        };

        params.progress_monitor.progress(&current_state, &context);

        let neighbour = params
            .annealing_strategy
            .neighbour(current_state, &context, &mut rng);
        let neighbour_energy = neighbour.energy();

        let accept = if neighbour_energy < current_energy {
            true
        } else {
            let acceptance_probability =
                (-(neighbour_energy - current_energy) / context.environment.temperature).exp();
            rng.gen::<f64>() < acceptance_probability
        };

        if accept {
            current_state = neighbour.accept();
            current_energy = neighbour_energy;
            if current_energy < context.best_energy {
                if let Some(solution) = current_state.as_solution() {
                    best_solution = Some(solution);
                    context.best_energy = current_energy;
                }
            }
        } else {
            current_state = neighbour.revert();
        }

        iterations += 1;
    }
}

pub fn run_annealing<S, AS, ES, SI, PM, RR, TR>(
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
    let threads = if params.threads > 0 {
        params.threads
    } else {
        std::thread::available_parallelism()
            .expect("Failed to get the available parallelism")
            .get()
    };
    let mut root_rng = params.root_rng;
    let annealing_strategy = &params.annealing_strategy;
    let progress_monitor = &params.progress_monitor;

    let thread_results: Vec<ThreadAnnealingResult<_>> = std::thread::scope(|scope| {
        let mut handles = Vec::new();
        let mut senders = Vec::new();

        for thread_id in 0..threads {
            let (sender, receiver) = mpsc::sync_channel(1);
            let initial_state = params
                .state_initializer
                .generate_initial_state(&mut root_rng);
            let thread_rng = TR::from_rng(&mut root_rng).unwrap();
            let handle = scope.spawn(move || {
                let thread_params = ThreadAnnealingParams {
                    thread_id,
                    annealing_strategy,
                    initial_state,
                    progress_monitor,
                    rng: thread_rng,
                    environment_receiver: receiver,
                };
                run_annealing_thread(thread_params)
            });
            handles.push(handle);
            senders.push(sender);
        }

        while let Some(environment) = params.environment_strategy.environment() {
            for sender in &senders {
                sender.try_send(environment).ok();
            }
            // TODO: Make the interruption period configurable.
            sleep(Duration::from_millis(100));
        }

        drop(senders);

        handles
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .collect()
    });

    let mut result = AnnealingResult {
        best_solution: None,
        best_energy: f64::INFINITY,
        final_states: Vec::new(),
        total_iterations: 0,
    };
    for thread_result in thread_results {
        if let Some(solution) = thread_result.best_solution {
            if thread_result.best_energy < result.best_energy {
                result.best_solution = Some(solution);
                result.best_energy = thread_result.best_energy;
            }
        }
        result.final_states.push(thread_result.final_state);
        result.total_iterations += thread_result.iterations;
    }

    result
}
