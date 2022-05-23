use std::fmt::Debug;

use rayon::{
    current_num_threads,
    iter::{IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator},
};

use crate::{
    operators::mapping::Mapping,
    operators::neighbour_gen::NeighbourGenerator,
    operators::{
        evaluation::Evaluation, initialisation::InitPop, solution::Constraint, solution::Solution,
    },
    utilities::nds::NonDominatedSet,
};

use super::{get_best, get_ref_points, get_weights, norm_tchebycheff};

pub fn run<
    X,
    Init: InitPop<X>,
    Map: Mapping<X> + Sync,
    NeighbourGen: NeighbourGenerator<X> + Sync + Clone,
    Eval: Evaluation + Sync + Clone,
>(
    init_pop: &Init,
    mapping: &Map,
    evaluate: &Eval,
    neighbour_gen: NeighbourGen,
    target_weights: usize,
    max_evaluations: usize,
    num_obj: usize,
    print_epochs: bool,
    mut iteration_observer: impl FnMut(usize, &Vec<Solution<X>>),
) where
    X: Clone + Debug + Sync + Send,
{
    let num_threads = current_num_threads();

    // Evaluate initial pop
    let weight_vectors = get_weights(target_weights, num_obj);
    let num_weights = weight_vectors.len();

    let num_epochs = (num_weights as f64 / num_threads as f64).ceil() as usize;

    let mut init_pop = init_pop.apply(num_weights);

    init_pop.par_iter_mut().for_each(|ind| {
        let routes = mapping.apply(&ind);
        ind.objectives = evaluate.evaluate_ind(&routes)
    });

    iteration_observer(0, &init_pop);

    let mut total_archive = NonDominatedSet::new(false);
    for ind in init_pop {
        total_archive.try_push(ind);
    }

    let remaining_evaluations = max_evaluations - num_weights;
    let num_evals_per_thread = remaining_evaluations / num_weights;

    let mut evaluations = num_weights; // Only used for the file name of the ouput

    for curr_epoch in 0..num_epochs {
        let (ref_point, nadir_point) = get_ref_points(&total_archive.get_raw(), num_obj);

        let start = curr_epoch * num_threads;
        let end = min(start + num_threads, num_weights);

        let epoch_archives: Vec<NonDominatedSet<X>> = weight_vectors[start..end]
            .par_iter()
            .map(|wv| {
                let mut neighbour_gen = neighbour_gen.clone();
                let evaluate = evaluate.clone();

                // Pick the best starting individual for the current weight
                let raw_archive = total_archive.get_raw();
                let (best_idx, _, _) = get_best(&raw_archive, &wv, &ref_point, &nadir_point);
                let mut best_ind = raw_archive[best_idx].clone();

                // Create archives
                let mut weight_archive = NonDominatedSet::new(false);
                weight_archive.try_push(best_ind.clone());

                let mut evals = 0;
                while evals < num_evals_per_thread {
                    let mut neighbour = neighbour_gen.apply(&best_ind);
                    let routes = mapping.apply(&neighbour);
                    neighbour.objectives = evaluate.evaluate_ind(&routes);
                    evals += 1;

                    weight_archive.try_push(neighbour.clone());

                    let curr_obj = &best_ind.objectives;
                    let neighbour_obj = &neighbour.objectives;

                    let accept = match (curr_obj, neighbour_obj) {
                        (Constraint::Feasible(_), Constraint::Infeasible(_)) => false,
                        (Constraint::Infeasible(_), Constraint::Feasible(_)) => true,
                        (Constraint::Infeasible(curr_vio), Constraint::Infeasible(neigh_vio)) => {
                            neigh_vio < curr_vio
                        }
                        (Constraint::Feasible(curr_objs), Constraint::Feasible(neigh_objs)) => {
                            // Compare based on fitness
                            let curr_tch =
                                norm_tchebycheff(&curr_objs, &wv, &ref_point, &nadir_point);
                            let neigh_tch =
                                norm_tchebycheff(&neigh_objs, &wv, &ref_point, &nadir_point);

                            neigh_tch <= curr_tch
                        }
                        _ => panic!("Undefined objectives: {:?} {:?}", curr_obj, neighbour_obj),
                    };

                    if accept {
                        best_ind = neighbour;
                    }
                }

                weight_archive
            })
            .collect();

        // Extend archive
        for set in epoch_archives {
            for solution in set.get_raw() {
                total_archive.try_push(solution.clone());
            }
        }

        evaluations += num_evals_per_thread * (end - start);

        if print_epochs {
            iteration_observer(evaluations, total_archive.get_raw());
        }
    }

    iteration_observer(evaluations, total_archive.get_raw());
}

fn min(x: usize, y: usize) -> usize {
    x.min(y)
}
