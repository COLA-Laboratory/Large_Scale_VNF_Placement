use rand::prelude::*;
use std::f64;

use crate::{
    operators::{
        crossover::Crossover,
        evaluation::Evaluation,
        initialisation::InitPop,
        mapping::Mapping,
        neighbour_gen::NeighbourGenerator,
        solution::{Constraint, Solution},
    },
    utilities::nds::NonDominatedSet,
};

use super::{get_ref_points, get_weights, norm_tchebycheff};

pub fn run<
    X: Clone,
    Init: InitPop<X>,
    Map: Mapping<X>,
    Mutate: NeighbourGenerator<X>,
    Cross: Crossover<X>,
    Eval: Evaluation,
>(
    init_pop: &Init,
    mapping: &Map,
    evaluate: &Eval,
    mut mutation: Mutate,
    crossover: &Cross,
    num_obj: usize,
    max_evaluations: usize,
    pop_size: usize,
    num_neighbours: usize,
    mut iteration_observer: impl FnMut(usize, &Vec<Solution<X>>),
) {
    let weight_vectors: Vec<Vec<f64>> = get_weights(pop_size, num_obj);
    let neighbours: Vec<Vec<usize>> = get_neighbours(num_neighbours, &weight_vectors, num_obj);

    let num_weights = weight_vectors.len();

    // Initial population
    let mut external_pop = NonDominatedSet::new(false);
    let mut weights_pop = init_pop.apply(num_weights);

    for ind in weights_pop.iter_mut() {
        let routes = mapping.apply(&ind);
        ind.objectives = evaluate.evaluate_ind(&routes);

        external_pop.try_push(ind.clone());
    }

    iteration_observer(0, &external_pop.get_raw());

    let mut evaluations = weights_pop.len();

    let (mut ref_point, mut nadir_point) = get_ref_points(&weights_pop, num_obj);

    let mut rng = rand::thread_rng();

    while evaluations < max_evaluations {
        for i in 0..num_weights {
            if evaluations >= max_evaluations {
                break;
            }

            let n_a = rng.gen_range(0..num_neighbours);
            let n_b = rng.gen_range(0..num_neighbours);

            let ind_a = neighbours[i][n_a];
            let ind_b = neighbours[i][n_b];

            let children = crossover.apply(&weights_pop[ind_a], &weights_pop[ind_b]);
            let mut children: Vec<Solution<X>> = children
                .into_iter()
                .map(|ind| mutation.apply(&ind))
                .collect();

            for ind in children.iter_mut() {
                let routes = mapping.apply(&ind);
                ind.objectives = evaluate.evaluate_ind(&routes)
            }

            evaluations = evaluations + children.len();

            for n_id in &neighbours[i] {
                for child in &children {
                    let wv = &weight_vectors[*n_id];
                    let curr = &weights_pop[*n_id];

                    let accept = match (&curr.objectives, &child.objectives) {
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
                        _ => panic!(
                            "Undefined objectives: {:?} {:?}",
                            child.objectives, curr.objectives
                        ),
                    };

                    if accept {
                        weights_pop[*n_id] = child.clone();
                    }
                }
            }

            let (r, n) = get_ref_points(&weights_pop, num_obj);
            ref_point = r;
            nadir_point = n;
        }

        // Update external pop
        for idx in 0..weights_pop.len() {
            external_pop.try_push(weights_pop[idx].clone());
        }

        // iteration_observer(evaluations, &external_pop);
    }

    iteration_observer(evaluations, &external_pop.get_raw());
}

fn get_neighbours(
    num_neighbours: usize,
    weight_vectors: &Vec<Vec<f64>>,
    num_obj: usize,
) -> Vec<Vec<usize>> {
    let mut neighbours: Vec<Vec<usize>> =
        vec![Vec::with_capacity(num_neighbours); weight_vectors.len()];

    for i in 0..weight_vectors.len() {
        let mut dists: Vec<(usize, f64)> = Vec::with_capacity(weight_vectors.len());

        for j in 0..weight_vectors.len() {
            // Calculate distance between vector i and j
            let mut dist = 0.0;
            for k in 0..num_obj {
                dist = dist + (weight_vectors[i][k] - weight_vectors[j][k]).powf(2.0);
            }
            dist = dist.sqrt();

            dists.push((j, dist));
        }

        dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let neighbour_idxs: Vec<usize> = dists[0..num_neighbours].iter().map(|x| x.0).collect();
        neighbours[i] = neighbour_idxs;
    }

    neighbours
}

pub fn get_h(pop_size: usize) -> usize {
    let pop_size = pop_size as i32;

    let pop_to_h = vec![
        28, 36, 45, 55, 66, 78, 91, 105, 120, 136, 153, 171, 190, 210, 231, 253, 276, 300, 325,
        351, 378, 406, 435, 465, 496, 528, 561, 595,
    ];

    let mut dist = pop_size - pop_to_h[0];

    let mut i = 0;

    loop {
        let c_dist = (pop_size - pop_to_h[i]).abs();
        if c_dist < dist {
            dist = c_dist;
        }

        if c_dist > dist {
            break;
        }

        i = i + 1;
    }

    i + 5
}

pub fn calc_weights(h: usize, num_obj: usize) -> Vec<Vec<f64>> {
    let mut weights = Vec::new();
    for i in 0..=h {
        for j in 0..=h {
            if i + j <= h {
                let k = h - i - j;
                let mut weight = Vec::with_capacity(num_obj);

                weight.push(i as f64 / h as f64);
                weight.push(j as f64 / h as f64);
                weight.push(k as f64 / h as f64);

                // Normalise weight
                let mag = weight.iter().map(|w| w.powf(2.0)).sum::<f64>().sqrt();
                let weight = weight.into_iter().map(|w| w / mag).collect();

                weights.push(weight);
            }
        }
    }

    weights
}