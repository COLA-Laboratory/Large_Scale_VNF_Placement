use crate::operators::solution::{Constraint, Solution};

pub mod aw_ppls;
pub mod ibea;
pub mod moead;
pub mod mw_ppls;
pub mod nsgaii;
pub mod ppls_simple;
pub mod pplsd;

fn get_best<'a, X: Clone>(
    pop: &'a Vec<Solution<X>>,
    wv: &Vec<f64>,
    ref_point: &Vec<f64>,
    nadir_point: &Vec<f64>,
) -> (usize, f64, usize) {
    let mut best_ind = 0;
    let mut min_dist = std::f64::INFINITY;
    let mut min_infeasible = std::usize::MAX;

    for (i, ind) in pop.iter().enumerate() {
        match (&ind.objectives, &pop[best_ind].objectives) {
            (Constraint::Feasible(ind_objectives), _) => {
                let dist = norm_tchebycheff(&ind_objectives, &wv, &ref_point, &nadir_point);

                if dist < min_dist {
                    min_dist = dist;
                    best_ind = i;
                    min_infeasible = 0;
                }
            }
            (Constraint::Infeasible(ind_constraint), Constraint::Infeasible(_)) => {
                if *ind_constraint < min_infeasible {
                    min_infeasible = *ind_constraint;
                    best_ind = i;
                }
            }
            (Constraint::Infeasible(_), Constraint::Feasible(_)) => {
                // do nothing
            }
            _ => panic!("One or more objectives undefined"),
        }
    }

    (best_ind, min_dist, min_infeasible)
}

fn get_weights(pop_size: usize, num_obj: usize) -> Vec<Vec<f64>> {
    if num_obj == 2 {
        let mut weights = Vec::with_capacity(pop_size);

        for i in 0..pop_size {
            let a = (i as f64) / (pop_size - 1) as f64;

            let weight = vec![a, 1.0 - a];

            let mag = weight.iter().map(|w| w.powf(2.0)).sum::<f64>().sqrt();
            let weight = weight.into_iter().map(|w| w / mag).collect();

            weights.push(weight);
        }

        return weights;
    } else if num_obj == 3 {
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

        let h = i + 5;

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

        return weights;
    } else {
        unimplemented!()
    }
}

fn norm_tchebycheff(
    objectives: &Vec<f64>,
    weights: &Vec<f64>,
    ref_point: &Vec<f64>,
    nadir_point: &Vec<f64>,
) -> f64 {
    let mut max = std::f64::MIN;

    for i in 0..objectives.len() {
        let dist =
            weights[i] * ((objectives[i] - ref_point[i]) / (nadir_point[i] - ref_point[i])).abs();

        if dist > max {
            max = dist;
        }
    }
    max
}

pub fn get_ref_points<X>(population: &Vec<Solution<X>>, num_obj: usize) -> (Vec<f64>, Vec<f64>) {
    let mut ref_point = vec![std::f64::MAX; num_obj];
    let mut nadir_point = vec![std::f64::MIN; num_obj];

    for ind in population {
        if !ind.objectives.is_feasible() {
            continue;
        }

        let obj = ind.objectives.unwrap();

        for i in 0..num_obj {
            if obj[i] < ref_point[i] {
                ref_point[i] = obj[i];
            }

            if obj[i] > nadir_point[i] {
                nadir_point[i] = obj[i];
            }
        }
    }

    (ref_point, nadir_point)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operators::solution::{Constraint, Solution};

    #[test]
    pub fn test_get_best() {
        let mut solutions = vec![Solution::new(vec![0]); 4];

        let wv = vec![1.0, 1.0];
        let ref_point = vec![0.0, 0.0];
        let nadir_point = vec![1.0, 1.0];

        // All feasible
        solutions[0].objectives = Constraint::Feasible(vec![0.0, 1.0]);
        solutions[1].objectives = Constraint::Feasible(vec![0.25, 0.75]);
        solutions[2].objectives = Constraint::Feasible(vec![0.5, 0.5]);
        solutions[3].objectives = Constraint::Feasible(vec![0.75, 0.25]);

        let (af_id, af_dist, af_cv) = get_best(&solutions, &wv, &ref_point, &nadir_point);

        // Mixed
        solutions[0].objectives = Constraint::Infeasible(3);
        solutions[1].objectives = Constraint::Feasible(vec![0.45, 0.55]);
        solutions[2].objectives = Constraint::Infeasible(1);
        solutions[3].objectives = Constraint::Feasible(vec![0.75, 0.25]);

        let (m_id, m_dist, m_cv) = get_best(&solutions, &wv, &ref_point, &nadir_point);

        // All infeasible
        solutions[0].objectives = Constraint::Infeasible(3);
        solutions[1].objectives = Constraint::Infeasible(4);
        solutions[2].objectives = Constraint::Infeasible(5);
        solutions[3].objectives = Constraint::Infeasible(4);

        let (if_id, if_dist, if_cv) = get_best(&solutions, &wv, &ref_point, &nadir_point);

        assert_eq!(af_id, 2);
        assert_eq!(af_dist, 0.5);
        assert_eq!(af_cv, 0);

        assert_eq!(m_id, 1);
        assert_eq!(m_dist, 0.55);
        assert_eq!(m_cv, 0);

        assert_eq!(if_id, 0);
        assert_eq!(if_dist, std::f64::INFINITY);
        assert_eq!(if_cv, 3);
    }

    #[test]
    pub fn test_tchebycheff() {
        let weights = vec![1.0, 1.0];
        let ref_point = vec![0.0, 0.0];
        let nadir_point = vec![2.0, 2.0];

        let a = vec![0.5, 0.5];
        let b = vec![0.0, 1.0];
        let c = vec![0.5, 2.0];
        let d = vec![2.0, 0.0];
        let e = vec![0.1, 0.1];

        let a_dist = norm_tchebycheff(&a, &weights, &ref_point, &nadir_point);
        let b_dist = norm_tchebycheff(&b, &weights, &ref_point, &nadir_point);
        let c_dist = norm_tchebycheff(&c, &weights, &ref_point, &nadir_point);
        let d_dist = norm_tchebycheff(&d, &weights, &ref_point, &nadir_point);
        let e_dist = norm_tchebycheff(&e, &weights, &ref_point, &nadir_point);

        assert!(a_dist < b_dist && a_dist < c_dist && a_dist < d_dist && a_dist > e_dist);
        assert!(b_dist > a_dist && b_dist < c_dist && b_dist < d_dist && a_dist > e_dist);
        assert!(c_dist > a_dist && c_dist > b_dist && c_dist == d_dist && a_dist > e_dist);
        assert!(d_dist > a_dist && d_dist > b_dist && d_dist == c_dist && a_dist > e_dist);
        assert!(e_dist < a_dist && e_dist < b_dist && e_dist < c_dist && e_dist < d_dist);
    }

    #[test]
    pub fn test_get_ref_point() {
        let mut pop: Vec<Solution<f64>> = (0..5)
            .into_iter()
            .map(|i| Solution::new(vec![i as f64]))
            .collect();

        pop[0].objectives = Constraint::Feasible(vec![2.0, 2.0]);
        pop[1].objectives = Constraint::Feasible(vec![1.8, 2.2]);
        pop[2].objectives = Constraint::Feasible(vec![1.6, 3.1]);
        pop[3].objectives = Constraint::Feasible(vec![3.0, 3.0]);
        pop[4].objectives = Constraint::Feasible(vec![2.1, 1.9]);

        let (utopia, nadir) = get_ref_points(&pop, 2);

        assert_eq!(utopia, vec![1.6, 1.9]);
        assert_eq!(nadir, vec![3.0, 3.1]);
    }
}
