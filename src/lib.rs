use rand::prelude::*;
use std::{thread, time::Instant};
use thousands::Separable;

pub struct AnnealingOptions {
    pub time_limit: f64,
    pub limit_temp: f64,
    pub restart: usize,
    pub threads: usize,
    pub silent: bool,
    pub header: String,
}

pub struct AnnealingResult<A: Annealer> {
    pub score: f64,
    pub iterations: usize,
    pub state: Option<A::State>,
}

pub trait Annealer {
    type State: Clone + Send + Sync;
    type Move;

    fn init_state(&self, rng: &mut impl Rng) -> Self::State;
    fn start_temp(&self, init_score: f64) -> f64;

    fn is_done(&self, _score: f64) -> bool {
        false
    }

    fn eval(
        &self,
        state: &Self::State,
        progress_ratio: f64,
        best_score: f64,
        valid_best_score: f64,
    ) -> (f64, Option<f64>);

    fn neighbour(
        &self,
        state: &mut Self::State,
        rng: &mut impl Rng,
        progress_ratio: f64,
    ) -> Self::Move;

    fn apply(&self, state: &mut Self::State, mov: &Self::Move);
    fn unapply(&self, state: &mut Self::State, mov: &Self::Move);

    fn apply_and_eval(
        &self,
        state: &mut Self::State,
        mov: &Self::Move,
        progress_ratio: f64,
        best_score: f64,
        valid_best_score: f64,
        _prev_score: f64,
    ) -> (f64, Option<f64>) {
        self.apply(state, mov);
        self.eval(state, progress_ratio, best_score, valid_best_score)
    }
}

pub fn annealing<A: 'static + Annealer + Sync>(
    annealer: &A,
    opt: &AnnealingOptions,
    seed: u64,
) -> AnnealingResult<A> {
    assert!(opt.threads > 0);

    if opt.threads == 1 {
        do_annealing(None, annealer, opt, seed)
    } else {
        let mut rng = StdRng::seed_from_u64(seed);

        let res = thread::scope(|s| {
            let mut ths = vec![];

            for i in 0..opt.threads {
                let tl_seed = rng.gen();
                ths.push(s.spawn(move || do_annealing(Some(i), annealer, opt, tl_seed)));
            }

            ths.into_iter()
                .map(|th| th.join().unwrap())
                .collect::<Vec<_>>()
        });

        if !opt.silent {
            eprintln!("===== results =====");
            for (i, r) in res.iter().enumerate() {
                eprintln!("[{}]: score: {}", i, r.score);
            }
        }

        // res.into_iter()
        //     .filter_map(|th| th)
        //     .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        let mut ret = AnnealingResult {
            iterations: 0,
            score: f64::INFINITY,
            state: None,
        };

        for r in res {
            ret.iterations += r.iterations;

            if let Some(s) = r.state {
                if r.score < ret.score {
                    ret.score = r.score;
                    ret.state = Some(s);
                }
            }
        }

        ret
    }
}

fn do_annealing<A: Annealer>(
    thread_id: Option<usize>,
    annealer: &A,
    opt: &AnnealingOptions,
    seed: u64,
) -> AnnealingResult<A> {
    let mut rng = SmallRng::seed_from_u64(seed);

    let mut state = annealer.init_state(&mut rng);
    let (mut cur_score, init_correct_score) =
        annealer.eval(&mut state, 0.0, f64::INFINITY, f64::INFINITY);

    let mut best_score = cur_score;

    let mut valid_best_score = f64::INFINITY;
    let mut valid_best_ans = if let Some(score) = init_correct_score {
        valid_best_score = score;
        Some(state.clone())
    } else {
        None
    };

    macro_rules! progress {
            ($($arg:expr),*) => {
                if !opt.silent {
                    if let Some(tid) = thread_id {
                        eprint!("[{:02}] ", tid);
                    }
                    eprint!("{}", opt.header);
                    eprintln!($($arg),*);
                }
            };
        }

    progress!("Initial score: {}", cur_score);

    let mut restart_cnt = 0;

    let t_max = annealer.start_temp(cur_score);
    let t_min = opt.limit_temp;

    let mut timer = Instant::now();
    let time_limit = opt.time_limit;

    let mut temp = t_max;
    let mut progress_ratio = 0.0;
    let mut prev_heart_beat = timer.elapsed();

    let mut iters = 0;

    for i in 0.. {
        if i % 100 == 0 {
            progress_ratio = timer.elapsed().as_secs_f64() / time_limit;
            if progress_ratio >= 1.0 {
                restart_cnt += 1;
                if restart_cnt >= opt.restart {
                    progress!(
                        "{} iteration processed, {:.2} iter/s",
                        i,
                        i as f64 / time_limit
                    );
                    break;
                }
                progress!("Restarting... {}/{}", restart_cnt, opt.restart);

                timer = Instant::now(); // - Duration::from_secs_f64(time_limit / 2.0);
            }

            temp = t_max * (t_min / t_max).powf(progress_ratio);

            if (timer.elapsed() - prev_heart_beat).as_secs_f64() >= 10.0 {
                progress!(
                    "best = {:>16}, best valid = {:>16}, cur = {:>16}, temp = {:>16}, progress: {:6.2}% ⛔",
                    format!("{:.1}", best_score).separate_with_commas(),
                    format!("{:.1}", valid_best_score).separate_with_commas(),
                    format!("{:.1}", cur_score).separate_with_commas(),
                    format!("{:.1}", temp).separate_with_commas(),
                    progress_ratio * 100.0
                );
                prev_heart_beat = timer.elapsed();
            }
        }

        iters += 1;

        let mov = annealer.neighbour(&mut state, &mut rng, progress_ratio);

        let (new_score, new_correct_score) = annealer.apply_and_eval(
            &mut state,
            &mov,
            progress_ratio,
            best_score,
            valid_best_score,
            cur_score,
        );

        let mut best_valid_updated = false;
        if let Some(new_correct_score) = new_correct_score {
            if new_correct_score < valid_best_score {
                if valid_best_score - new_correct_score > 1e-6 {
                    best_valid_updated = true;
                }
                valid_best_score = new_correct_score;
                valid_best_ans = Some(state.clone());
            }
        }

        let mut best_updated = false;

        if new_score <= cur_score
            || rng.gen::<f64>() <= ((cur_score - new_score) as f64 / temp).exp()
        {
            cur_score = new_score;

            if cur_score < best_score {
                if best_score - cur_score > 1e-6 {
                    best_updated = true;
                }

                best_score = cur_score;
            }

            if annealer.is_done(cur_score) {
                break;
            }
        } else {
            annealer.unapply(&mut state, &mov);
        }

        if best_updated || best_valid_updated {
            progress!(
                "best = {:>16}, best valid = {:>16}, cur = {:>16}, temp = {:>16}, progress: {:6.2}% {}",
                format!("{:.1}", best_score).separate_with_commas(),
                format!("{:.1}", valid_best_score).separate_with_commas(),
                format!("{:.1}", cur_score).separate_with_commas(),
                format!("{:.1}", temp).separate_with_commas(),
                progress_ratio * 100.0,
                if best_valid_updated { "✅" } else { "🐴" }
            );
            prev_heart_beat = timer.elapsed();
        }
    }

    AnnealingResult {
        iterations: iters,
        score: valid_best_score,
        state: valid_best_ans,
    }
}

// pub struct AnnealingResult<A: Annealer> {
//     pub score: f64,
//     pub iterations: usize,
//     pub state: Option<A::State>,
// }

// #[derive(Clone)]
// pub struct AnnealingOptions {
//     pub swap_interval: Option<f64>,
//     pub time_limit: f64,
//     pub limit_temp: f64,
//     pub restart: usize,
//     pub threads: usize,
//     pub silent: bool,
//     pub header: String,
// }

// pub trait Annealer {
//     type State: Clone + Send + Sync;
//     type Move;

//     fn init_state(&self, rng: &mut impl Rng) -> Self::State;
//     fn start_temp(&self, init_score: f64) -> f64;

//     fn is_done(&self, _score: f64) -> bool {
//         false
//     }

//     fn eval(
//         &self,
//         state: &Self::State,
//         best_score: f64,
//         valid_best_score: f64,
//     ) -> (f64, f64, bool);

//     fn neighbour(
//         &self,
//         state: &mut Self::State,
//         rng: &mut impl Rng,
//         progress_ratio: f64,
//     ) -> Self::Move;

//     fn apply(&self, state: &mut Self::State, mov: &Self::Move);
//     fn unapply(&self, state: &mut Self::State, mov: &Self::Move);

//     fn apply_and_eval(
//         &self,
//         state: &mut Self::State,
//         mov: &Self::Move,
//         best_score: f64,
//         valid_best_score: f64,
//         _prev_score: f64,
//     ) -> (f64, f64, bool) {
//         self.apply(state, mov);
//         self.eval(state, best_score, valid_best_score)
//     }
// }

// pub fn annealing<A: 'static + Annealer + Sync>(
//     annealer: &A,
//     opt: &AnnealingOptions,
//     seed: u64,
// ) -> AnnealingResult<A> {
//     assert!(opt.threads > 0);

//     let mut rng = StdRng::seed_from_u64(seed);

//     let mut states = (0..opt.threads)
//         .map(|thread_id| {
//             let seed = rng.gen();
//             let temp_range = if opt.swap_interval.is_some() {
//                 let start_temp = (thread_id + 1) as f64 * (0.5 / opt.threads as f64);
//                 start_temp..start_temp + 0.5
//             } else {
//                 0.0..1.0
//             };
//             AnnealingState::new(annealer, opt, seed, temp_range)
//         })
//         .collect::<Vec<_>>();

//     let mut cur_scores = vec![f64::INFINITY; opt.threads];

//     let phases = if let Some(swap_interval) = opt.swap_interval {
//         (opt.time_limit / swap_interval).round() as u64
//     } else {
//         1
//     };

//     for phase in 0..phases {
//         let cur_limit = (phase + 1) as f64 / phases as f64 * opt.time_limit;
//         thread::scope(|s| {
//             states.iter_mut().for_each(|state| {
//                 let _ = s.spawn(|| state.run(annealer, cur_limit));
//             });
//         });

//         for i in 0..opt.threads {
//             let score = states[i].valid_best_score;

//             let update = if score < cur_scores[i] {
//                 cur_scores[i] = score;
//                 true
//             } else {
//                 false
//             };

//             let temp = states[i].temp();
//             if !opt.silent {
//                 eprintln!(
//                     "[{i:02}] {} best = {:12.3} (actual = {:12.3}), cur = {:12.3}, temp = {:12.3}, progress: {:6.2}% {}",
//                     opt.header,
//                     states[i].best_score,
//                     states[i].valid_best_score,
//                     states[i].cur_score,
//                     temp,
//                     cur_limit / opt.time_limit * 100.0,
//                     if update { "✅" } else { "🐴" }
//                 );
//             }
//         }

//         // Swap state
//         if opt.swap_interval.is_some() {
//             let mut swapped = vec![];

//             for i in ((phase as usize % 2)..opt.threads - 1).step_by(2) {
//                 let (x, y) = states.split_at_mut(i + 1);
//                 let x = &mut x[i];
//                 let y = &mut y[0];

//                 let tx = x.temp();
//                 let ty = y.temp();

//                 let err = y.cur_score - x.cur_score;
//                 let tdiff = ty - tx;

//                 if tdiff * err < 0.0 || rng.gen_bool((-(tdiff * err) / (tx * ty)).exp()) {
//                     swap(&mut x.cur_score, &mut y.cur_score);
//                     swap(&mut x.state, &mut y.state);
//                     swapped.push((i, i + 1));
//                 }
//             }

//             if !opt.silent {
//                 eprintln!(
//                     "# Phase [{phase}]: Swapped states: {}",
//                     swapped
//                         .into_iter()
//                         .map(|(i, j)| format!("{i} <-> {j}"))
//                         .collect::<Vec<_>>()
//                         .join(", ")
//                 );
//             }
//         }
//     }

//     let res = (0..opt.threads)
//         .map(|i| states[i].result())
//         .collect::<Vec<_>>();

//     if !opt.silent {
//         for i in 0..opt.threads {
//             eprintln!(
//                 "[{i}]: {} iteration processed, {:.2} iter/s",
//                 states[i].iteration_count,
//                 states[i].iteration_count as f64 / opt.time_limit
//             );
//         }

//         eprintln!("===== results =====");
//         for (i, r) in res.iter().enumerate() {
//             eprintln!(
//                 "[{}]: score: {}",
//                 i,
//                 r.as_ref().map_or(f64::INFINITY, |r| r.0)
//             );
//         }
//     }

//     let mut ret = AnnealingResult {
//         iterations: 0,
//         score: f64::INFINITY,
//         state: None,
//     };

//     for state in &states {
//         ret.iterations += state.iteration_count;
//         if let Some((score, state)) = state.result() {
//             if score < ret.score {
//                 ret.score = score;
//                 ret.state = Some(state);
//             }
//         }
//     }

//     // res.into_iter()
//     //     .filter_map(|r| r)
//     //     .min_by(|a, b| a.0.total_cmp(&b.0))

//     ret
// }

// struct AnnealingState<A: Annealer> {
//     rng: SmallRng,
//     t_max: f64,
//     t_min: f64,
//     temp_range: Range<f64>,

//     timer: SystemTime,
//     time_limit: f64,

//     state: A::State,
//     cur_score: f64,

//     best_score: f64,
//     best_annot_score: f64,
//     valid_best_score: f64,
//     valid_best_ans: Option<A::State>,

//     iteration_count: usize,
// }

// impl<A: Annealer> AnnealingState<A> {
//     fn new(annealer: &A, opt: &AnnealingOptions, seed: u64, temp_range: Range<f64>) -> Self {
//         let mut rng = SmallRng::seed_from_u64(seed);

//         let state = annealer.init_state(&mut rng);
//         let (cur_score, _, init_state_valid) =
//             annealer.eval(&state, f64::INFINITY, f64::INFINITY);

//         let t_max = annealer.start_temp(cur_score);
//         let t_min = opt.limit_temp;

//         Self {
//             rng,
//             t_max,
//             t_min,
//             temp_range,
//             timer: SystemTime::now(),
//             time_limit: opt.time_limit,
//             cur_score,
//             best_score: cur_score,
//             best_annot_score: cur_score,
//             valid_best_score: if init_state_valid {
//                 cur_score
//             } else {
//                 f64::INFINITY
//             },
//             valid_best_ans: if init_state_valid {
//                 Some(state.clone())
//             } else {
//                 None
//             },
//             state,
//             iteration_count: 0,
//         }
//     }

//     fn temp(&self) -> f64 {
//         let progress_ratio = self.timer.elapsed().unwrap().as_secs_f64() / self.time_limit;
//         let temp_ratio = self.temp_range.start
//             + (self.temp_range.end - self.temp_range.start) * progress_ratio;
//         self.t_max * (self.t_min / self.t_max).powf(temp_ratio)
//     }

//     fn run(&mut self, annealer: &A, cur_phase_limit: f64) {
//         // macro_rules! progress {
//         //     ($($arg:expr),*) => {
//         //         if !opt.silent {
//         //             eprint!("[{:02}] ", self.thread_id);
//         //             eprint!("{}", opt.header);
//         //             eprintln!($($arg),*);
//         //         }
//         //     };
//         // }

//         let mut temp = 0.0;
//         let mut progress_ratio = 0.0;
//         // let mut prev_heart_beat = self.timer.elapsed().unwrap();

//         for i in 0.. {
//             if i % 100 == 0 {
//                 let elapsed = self.timer.elapsed().unwrap().as_secs_f64();
//                 if elapsed > cur_phase_limit {
//                     break;
//                 }

//                 progress_ratio = elapsed / self.time_limit;

//                 let temp_ratio = self.temp_range.start
//                     + (self.temp_range.end - self.temp_range.start) * progress_ratio;

//                 temp = self.t_max * (self.t_min / self.t_max).powf(temp_ratio);

//                 // if (timer.elapsed().unwrap() - prev_heart_beat).as_secs_f64() >= 10.0 {
//                 //     progress!(
//                 //         "best = {:12.3}, valid best = {:12.3}, cur = {:12.3}, temp = {:12.3}, progress: {:6.2}% ⛔",
//                 //         best_score,
//                 //         valid_best_score,
//                 //         cur_score,
//                 //         temp,
//                 //         progress_ratio * 100.0
//                 //     );
//                 //     prev_heart_beat = timer.elapsed().unwrap();
//                 // }
//             }

//             self.iteration_count += 1;

//             let mov = annealer.neighbour(&mut self.state, &mut self.rng, progress_ratio);

//             let (new_score, new_annot, new_score_valid) = annealer.apply_and_eval(
//                 &mut self.state,
//                 &mov,
//                 self.best_score,
//                 self.valid_best_score,
//                 self.cur_score,
//             );

//             if new_annot < self.best_annot_score {
//                 self.best_annot_score = new_annot;
//             }

//             if new_score_valid && self.best_annot_score < self.valid_best_score {
//                 self.valid_best_score = self.best_annot_score;
//                 self.valid_best_ans = Some(self.state.clone());
//             }

//             if new_score <= self.cur_score
//                 || self.rng.gen::<f64>() <= ((self.cur_score - new_score) as f64 / temp).exp()
//             {
//                 self.cur_score = new_score;

//                 // let mut best_updated = false;
//                 // let mut best_valid_updated = false;

//                 if self.cur_score < self.best_score {
//                     // if self.best_score - self.cur_score > 1e-6 {
//                     //     best_updated = true;
//                     // }

//                     self.best_score = self.cur_score;
//                 }

//                 // if best_updated || best_valid_updated {
//                 // progress!(
//                 //     "best = {:12.3}, valid best = {:12.3}, cur = {:12.3}, temp = {:12.3}, progress: {:6.2}% {}",
//                 //     self.best_score,
//                 //     self.valid_best_score,
//                 //     self.cur_score,
//                 //     temp,
//                 //     progress_ratio * 100.0,
//                 //     if best_valid_updated { "✅" } else { "🐴" }
//                 // );
//                 // prev_heart_beat = self.timer.elapsed().unwrap();
//                 // }

//                 if annealer.is_done(self.cur_score) {
//                     break;
//                 }
//             } else {
//                 annealer.unapply(&mut self.state, &mov);
//             }
//         }
//     }

//     fn result(&self) -> Option<(f64, A::State)> {
//         if let Some(ans) = &self.valid_best_ans {
//             Some((self.valid_best_score, ans.clone()))
//         } else {
//             None
//         }
//     }
// }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}
