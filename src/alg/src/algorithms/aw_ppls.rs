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
    utilities::{math::euclidean_distance, nds::NonDominatedSet},
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
    num_epochs: usize,
    max_evaluations: usize,
    num_obj: usize,
    mut iteration_observer: impl FnMut(usize, &Vec<Solution<X>>),
) where
    X: Clone + Debug + Sync + Send,
{
    // Parallel Pareto Simulated Annealing
    let num_threads = current_num_threads();

    // Evaluate initial pop
    let mut init_pop = init_pop.apply(num_epochs * num_threads);

    init_pop.par_iter_mut().for_each(|ind| {
        let routes = mapping.apply(&ind);
        ind.objectives = evaluate.evaluate_ind(&routes)
    });

    let remaining_evaluations = max_evaluations - init_pop.len();
    let num_evals_per_epoch = remaining_evaluations / num_epochs;
    let num_evals_per_weight = num_evals_per_epoch / num_threads;

    let mut total_archive = NonDominatedSet::new(false);
    for ind in init_pop {
        total_archive.try_push(ind);
    }

    let mut weight_vectors = get_weights(num_threads, num_obj);

    for i in 0..num_epochs {
        let (ref_point, nadir_point) = get_ref_points(&total_archive.get_raw(), num_obj);

        let epoch_archives: Vec<NonDominatedSet<X>> = weight_vectors
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
                while evals < num_evals_per_weight {
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
                            curr_vio > neigh_vio
                        }
                        (Constraint::Feasible(curr_objs), Constraint::Feasible(neigh_objs)) => {
                            // Compare based on fitness
                            let curr_tch =
                                norm_tchebycheff(&curr_objs, &wv, &ref_point, &nadir_point);
                            let neigh_tch =
                                norm_tchebycheff(&neigh_objs, &wv, &ref_point, &nadir_point);

                            curr_tch > neigh_tch
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

        update_weight_vectors(&mut weight_vectors, &total_archive.get_raw(), num_obj);

        iteration_observer(i * num_evals_per_epoch, total_archive.get_raw());
    }
}

fn update_weight_vectors<X>(
    weight_vectors: &mut Vec<Vec<f64>>,
    total_archive: &Vec<Solution<X>>,
    num_obj: usize,
) {
    let mut sparsities = Vec::new();

    for i in 0..total_archive.len() {
        let solution = &total_archive[i];
        if solution.objectives.is_infeasible() {
            continue;
        }

        let sparsity = sparsity(solution, total_archive, num_obj);

        sparsities.push((i, sparsity));
    }

    sparsities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    for i in 0..weight_vectors.len() {
        let solution = &total_archive[sparsities[i].0];
        let objectives = solution.objectives.unwrap();

        weight_vectors[i] = ws_transformation(objectives);
    }
}

fn sparsity<X>(solution: &Solution<X>, archive: &Vec<Solution<X>>, num_obj: usize) -> f64 {
    // Assumes solution is feasible
    if solution.objectives.is_infeasible() {
        panic!("Solution provided to sparsity calculation is infeasible");
    }

    let mut dists = Vec::with_capacity(archive.len());

    for other in archive {
        if other.objectives.is_infeasible() {
            continue;
        }

        let dist = euclidean_distance(&solution.objectives.unwrap(), &other.objectives.unwrap());
        dists.push(dist);
    }

    // Sort descending
    dists.sort_by(|a, b| b.partial_cmp(a).unwrap());

    let mut sparsity = 1.0;
    for i in 0..num_obj {
        sparsity = sparsity * dists[i];
    }

    sparsity
}

fn ws_transformation(vector: Vec<f64>) -> Vec<f64> {
    let sum: f64 = vector.iter().map(|e| 1.0 / e).sum();
    let transformed = vector.iter().map(|e| (1.0 / e) / sum).collect();

    transformed
}
