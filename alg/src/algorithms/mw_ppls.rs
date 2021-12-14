use std::fmt::Debug;

use rayon::iter::{IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator};

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
    num_weights: usize,
    max_evaluations: usize,
    num_epochs: usize,
    per_ind_evaluations: usize,
    num_obj: usize,
    mut iteration_observer: impl FnMut(usize, &Vec<Solution<X>>),
    run_weight_update: bool,
) where
    X: Clone + Debug + Sync + Send,
{
    // A modified version of PPLS to use repulsive weights

    // Evaluate initial pop
    let weight_vectors = get_weights(num_weights, num_obj);
    let mut init_pop = init_pop.apply(num_weights);

    init_pop.par_iter_mut().for_each(|ind| {
        let routes = mapping.apply(&ind);
        ind.objectives = evaluate.evaluate_ind(&routes)
    });

    let mut total_archive = NonDominatedSet::new(false);
    for ind in init_pop {
        total_archive.try_push(ind);
    }

    let remaining_evaluations = max_evaluations - num_weights;
    let per_epoch_evaluations = remaining_evaluations / num_epochs;
    let per_weight_evaluations = per_epoch_evaluations / num_weights;

    for curr_epoch in 0..num_epochs {
        let (ref_point, nadir_point) = get_ref_points(&total_archive.get_raw(), num_obj);

        let epoch_archives: Vec<NonDominatedSet<X>> = weight_vectors
            .par_iter()
            .map(|wv| {
                let mut neighbour_gen = neighbour_gen.clone();

                let evaluate = evaluate.clone();

                // Pick the best starting individual for the current weight
                let raw_archive = total_archive.get_raw();
                let (best_idx, _, _) = get_best(&raw_archive, &wv, &ref_point, &nadir_point);
                let best_ind = &raw_archive[best_idx];

                // Create archives
                let mut weight_archive = NonDominatedSet::new(false);
                weight_archive.try_push(best_ind.clone());

                let mut unexplored_archive = Vec::new();
                unexplored_archive.push(best_ind.clone());

                let mut evaluations = 0;
                while evaluations < per_weight_evaluations && !unexplored_archive.is_empty() {
                    // Find the unexplored solution with the minimum tchbycheff distance
                    let (idx, best_dist, cnstr_violation) =
                        get_best(&unexplored_archive, &wv, &ref_point, &nadir_point);

                    let best_ind = unexplored_archive.swap_remove(idx);

                    // Generate and evaluate neighbouring solutions
                    let mut neighbours = Vec::new();
                    for _ in 0..per_ind_evaluations {
                        let mut ind = neighbour_gen.apply(&best_ind);

                        let routes = mapping.apply(&ind);
                        ind.objectives = evaluate.evaluate_ind(&routes);
                        evaluations += 1;

                        neighbours.push(ind);

                        if evaluations == per_weight_evaluations {
                            break;
                        }
                    }

                    let mut success = false;

                    for neighbour in &neighbours {
                        // Only consider feasible solutions at this stage since we usually break early
                        if neighbour.objectives.is_infeasible() {
                            continue;
                        }

                        let objectives = neighbour.objectives.unwrap();
                        let dist = norm_tchebycheff(&objectives, &wv, &ref_point, &nadir_point);

                        if dist < best_dist
                            && (is_in_region(&objectives, &wv, &weight_vectors)
                                || !any_in_region(&weight_archive, &wv, &weight_vectors))
                        {
                            weight_archive.try_push(neighbour.clone());

                            unexplored_archive.push(neighbour.clone());
                            success = true;

                            break;
                        }
                    }

                    if !success {
                        for neighbour in &neighbours {
                            // Accept infeasible solutions only if necessary
                            if let Constraint::Infeasible(neighbour_violation) =
                                neighbour.objectives
                            {
                                if neighbour_violation < cnstr_violation {
                                    weight_archive.try_push(neighbour.clone());
                                    unexplored_archive.push(neighbour.clone());
                                }

                                continue;
                            }

                            let objectives = neighbour.objectives.unwrap();

                            if is_in_region(&objectives, &wv, &weight_vectors)
                                || !any_in_region(&weight_archive, &wv, &weight_vectors)
                            {
                                let is_added = weight_archive.try_push(neighbour.clone());
                                if is_added {
                                    unexplored_archive.push(neighbour.clone());
                                }
                            }
                        }
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

        iteration_observer(curr_epoch * per_epoch_evaluations, total_archive.get_raw());
    }
}

fn is_in_region(solution: &Vec<f64>, weight: &Vec<f64>, other_weights: &Vec<Vec<f64>>) -> bool {
    let cmp_angle = angle(solution, weight);

    for oth_weight in other_weights {
        let oth_angle = angle(solution, oth_weight);

        if oth_angle < cmp_angle {
            return false;
        }
    }

    true
}

fn any_in_region<X>(
    solutions: &NonDominatedSet<X>,
    weight: &Vec<f64>,
    other_weights: &Vec<Vec<f64>>,
) -> bool {
    for solution in solutions.get_raw() {
        if solution.objectives.is_infeasible() {
            continue;
        }

        let objectives = solution.objectives.unwrap();

        if is_in_region(&objectives, weight, other_weights) {
            return true;
        }
    }

    false
}

fn angle(vec_a: &Vec<f64>, vec_b: &Vec<f64>) -> f64 {
    (dot_product(vec_a, vec_b) / (magnitude(vec_a) * magnitude(vec_b))).acos()
}

fn magnitude(vec_a: &Vec<f64>) -> f64 {
    vec_a.iter().map(|c| c.powf(2.0)).sum::<f64>().sqrt()
}

fn dot_product(vec_a: &Vec<f64>, vec_b: &Vec<f64>) -> f64 {
    vec_a.iter().zip(vec_b).map(|(a, b)| a * b).sum()
}
