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
    target_weights: usize,
    max_evaluations: usize,
    per_ind_evaluations: usize,
    num_obj: usize,
    mut iteration_observer: impl FnMut(usize, &Vec<Solution<X>>),
) where
    X: Clone + Debug + Sync + Send,
{
    // PPLS runs a modified pareto local search on different weight vectors in parallel.
    // Since the different threads do not communicate with each other, we execute *all* evaluations
    // for each weight vector before continuing to the next one. This ensures there isn't too much
    // moving about of memory

    // Evaluate initial pop
    let weight_vectors = get_weights(target_weights, num_obj);
    let num_weights = weight_vectors.len();

    let mut init_archive = init_pop.apply(num_weights);

    init_archive.par_iter_mut().for_each(|ind| {
        let routes = mapping.apply(&ind);
        ind.objectives = evaluate.evaluate_ind(&routes)
    });

    iteration_observer(0, &init_archive);

    let (ref_point, nadir_point) = get_ref_points(&init_archive, num_obj);

    let remaining_evaluations = max_evaluations - num_weights;
    let per_weight_evaluations = remaining_evaluations / num_weights;

    let total_archive: Vec<NonDominatedSet<X>> = weight_vectors
        .par_iter()
        .map(|wv| {
            let mut neighbour_gen = neighbour_gen.clone();

            let evaluate = evaluate.clone();

            // Pick the best starting individual for the current weight
            let (best_idx, _, _) = get_best(&init_archive, &wv, &ref_point, &nadir_point);
            let best_ind = &init_archive[best_idx];

            // Create archives
            let mut archive = NonDominatedSet::new(false);
            archive.try_push(best_ind.clone());

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
                            || !any_in_region(&archive, &wv, &weight_vectors))
                    {
                        archive.try_push(neighbour.clone());
                        unexplored_archive.push(neighbour.clone());
                        
                        success = true;

                        break;
                    }
                }

                if !success {
                    for neighbour in &neighbours {
                        // Accept infeasible solutions only if necessary
                        if let Constraint::Infeasible(neighbour_violation) = neighbour.objectives {
                            if neighbour_violation < cnstr_violation {
                                archive.try_push(neighbour.clone());
                                unexplored_archive.push(neighbour.clone());
                            }

                            continue;
                        }

                        let objectives = neighbour.objectives.unwrap();

                        if is_in_region(&objectives, &wv, &weight_vectors)
                            || !any_in_region(&archive, &wv, &weight_vectors)
                        {
                            let is_added = archive.try_push(neighbour.clone());
                            if is_added {
                                unexplored_archive.push(neighbour.clone());
                            }
                        }
                    }
                }
            }

            archive
        })
        .collect();

    // Merge non-dominated sets and report result
    let mut final_solutions = NonDominatedSet::new(false);
    for set in total_archive {
        for solution in set.get_raw() {
            final_solutions.try_push(solution.clone());
        }
    }

    iteration_observer(max_evaluations, final_solutions.get_raw());
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::math::round_to;

    #[test]
    pub fn test_in_region() {
        let ref_weight = vec![1.0, 1.0];
        let oth_weights = vec![vec![0.0, 1.0], vec![1.0, 0.0]];

        // Yes
        let a = vec![1.0, 1.0];
        let b = vec![0.5, 0.5];

        // No
        let c = vec![0.0, 1.0];
        let d = vec![0.2, 1.0];
        let e = vec![3.0, 1.0];

        let t_a = is_in_region(&a, &ref_weight, &oth_weights);
        let t_b = is_in_region(&b, &ref_weight, &oth_weights);
        let t_c = is_in_region(&c, &ref_weight, &oth_weights);
        let t_d = is_in_region(&d, &ref_weight, &oth_weights);
        let t_e = is_in_region(&e, &ref_weight, &oth_weights);

        assert!(t_a);
        assert!(t_b);
        assert!(!t_c);
        assert!(!t_d);
        assert!(!t_e);
    }

    #[test]
    pub fn test_angle() {
        let r = vec![1.0, 1.0];

        let a = vec![2.0, 1.0];
        let b = vec![2.0, 2.0];
        let c = vec![1.0, -1.0];
        let d = vec![0.0, 1.0];

        let ang_a = angle(&r, &a);
        let ang_b = angle(&r, &b);
        let ang_c = angle(&r, &c);
        let ang_d = angle(&r, &d);

        assert_eq!(round_to(ang_a, 3), 0.322);
        assert_eq!(round_to(ang_b, 3), 0.0);
        assert_eq!(round_to(ang_c, 3), 1.571);
        assert_eq!(round_to(ang_d, 3), 0.785);
    }

    #[test]
    pub fn test_magnitude() {
        let x = vec![1.0, 2.0, 3.0];

        let m = magnitude(&x);

        let t: f64 = 14.0;
        assert_eq!(m, t.sqrt());
    }

    #[test]
    pub fn test_dot_product() {
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![3.0, 5.0, 2.0];

        let dp = dot_product(&x, &y);

        assert_eq!(dp, 19.0);
    }
}
