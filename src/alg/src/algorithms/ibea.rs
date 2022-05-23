use std::f64;

use crate::operators::{
    crossover::Crossover, evaluation::Evaluation, initialisation::InitPop, mapping::Mapping,
    neighbour_gen::NeighbourGenerator, selection::TournamentSelection, solution::Solution,
};

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
    max_evaluations: usize,
    pop_size: usize,
    k: f64, // IBEA specific parameter
    mut iteration_observer: impl FnMut(usize, &Vec<Solution<X>>),
) {
    let mut pop = init_pop.apply(pop_size);
    let mut child_pop: Vec<Solution<X>> = Vec::with_capacity(pop_size);

    let mut evaluations;

    for ind in pop.iter_mut() {
        let routes = mapping.apply(&ind);
        ind.objectives = evaluate.evaluate_ind(&routes)
    }

    iteration_observer(0, &pop);

    evaluations = pop.len();

    loop {
        // Calculate solution fitness
        let mut fitnesses = vec![0.0; pop.len()];

        for i in 0..pop.len() {
            let obj_i = &pop[i].objectives;
            if obj_i.is_infeasible() {
                fitnesses[i] = std::f64::NEG_INFINITY;
                continue;
            }

            let mut fitness = 0.0;
            for j in 0..pop.len() {
                if i == j {
                    continue;
                }

                let obj_j = &pop[j].objectives;
                if obj_j.is_feasible() {
                    fitness += -(-add_eps_indicator(&obj_i.unwrap(), &obj_j.unwrap()) / k).exp();
                }
            }

            fitnesses[i] = fitness;
        }

        while pop.len() > pop_size {
            let worst_idx = get_worst_ind(&fitnesses);

            let worst_objs = &pop[worst_idx].objectives;
            if worst_objs.is_feasible() {
                // Update fitness values of the remaining individuals
                for i in 0..pop.len() {
                    let objs_i = &pop[i].objectives;
                    if objs_i.is_infeasible() {
                        continue;
                    }

                    fitnesses[i] +=
                        (-add_eps_indicator(&objs_i.unwrap(), &worst_objs.unwrap()) / k).exp()
                }
            }

            pop.remove(worst_idx);
            fitnesses.remove(worst_idx);
        }

        if evaluations >= max_evaluations {
            break;
        }

        // Generate offspring
        let ts = TournamentSelection::new(pop.len(), |x, y| fitnesses[x] > fitnesses[y]);

        while child_pop.len() < pop_size {
            let parent_one = ts.tournament(2);
            let parent_two = ts.tournament(2);

            let new_children = crossover.apply(&pop[parent_one], &pop[parent_two]);
            for child in new_children {
                let mut child = mutation.apply(&child);

                let routes = mapping.apply(&child);
                child.objectives = evaluate.evaluate_ind(&routes);

                evaluations = evaluations + 1;

                child_pop.push(child);
            }
        }

        pop.append(&mut child_pop);
    }

    iteration_observer(evaluations, &pop);
}

pub fn get_worst_ind(fitness: &Vec<f64>) -> usize {
    let mut min_idx = 0;
    let mut min_fit = f64::MAX;

    for (i, fit) in fitness.iter().enumerate() {
        if fit < &min_fit {
            min_idx = i;
            min_fit = *fit;
        }
    }

    min_idx
}

pub fn add_eps_indicator(a_objs: &Vec<f64>, b_objs: &Vec<f64>) -> f64 {
    let mut eps = f64::NEG_INFINITY;

    for i in 0..a_objs.len() {
        eps = f64::max(eps, b_objs[i] - a_objs[i]);
    }

    eps
}

pub fn norm_objectives(objectives: &Vec<f64>, max_obj: &Vec<f64>, min_obj: &Vec<f64>) -> Vec<f64> {
    objectives
        .iter()
        .enumerate()
        .map(|(i, x)| (x - min_obj[i]) / (max_obj[i] - min_obj[i]))
        .collect()
}
