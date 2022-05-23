use rand::{distributions::WeightedIndex, prelude::*};
use rand_distr::Beta;

use crate::operators::solution::Solution;
pub trait NeighbourGenerator<X> {
    fn apply(&mut self, solution: &Solution<X>) -> Solution<X>;
    fn update(&mut self, obj_one: f64, obj_two: f64);
}

// *** Common functions *** //
fn add_item<X: Clone>(solution: &mut Solution<Vec<X>>, items: &Vec<X>, weights: Vec<usize>) {
    let mut rng = rand::thread_rng();
    let dist = WeightedIndex::new(&weights).unwrap();

    let idx = rng.gen_range(0..solution.len());
    let rnd_item = dist.sample(&mut rng);

    solution[idx].push(items[rnd_item].clone());
}

fn remove_item<X: Clone>(solution: &mut Solution<Vec<X>>) {
    // Remove item
    let mut rng = rand::thread_rng();
    let mut ids: Vec<usize> = (0..solution.len()).collect();
    ids.shuffle(&mut rng);

    for id in ids {
        if !solution[id].is_empty() {
            let rn = rng.gen_range(0..solution[id].len());
            solution[id].swap_remove(rn);

            break;
        }
    }
}

// fn rem_item_by_type<X: Clone + PartialEq>(solution: &mut Solution<Vec<X>>, target: &X) {
//     // Remove item
//     let mut rng = rand::thread_rng();
//     let mut ids: Vec<usize> = (0..solution.len()).collect();
//     ids.shuffle(&mut rng);

//     for id in ids {
//         if !solution[id].is_empty() {
//             for i in 0..solution[id].len() {
//                 if &solution[id][i] == target {
//                     solution[id].swap_remove(i);
//                     break;
//                 }
//             }
//         }
//     }
// }

fn move_item<X>(solution: &mut Solution<Vec<X>>) {
    let rng = &mut thread_rng();
    let len = solution.len();

    loop {
        let src_server_id = rng.gen_range(0..len);
        let src_server = &mut solution[src_server_id];

        if src_server.is_empty() {
            continue;
        }

        let vm_id = rng.gen_range(0..src_server.len());

        let mut dest_server = src_server_id;
        while dest_server == src_server_id {
            dest_server = rng.gen_range(0..len);
        }

        let vnf = src_server.remove(vm_id);
        solution[dest_server].push(vnf);

        break;
    }
}

fn swap_servers<X: Clone>(solution: &mut Solution<Vec<X>>) {
    let rng = &mut thread_rng();

    // Swap server
    let a = rng.gen_range(0..solution.len());
    let b = rng.gen_range(0..solution.len());

    let temp = solution[a].clone();
    solution[a] = solution[b].clone();
    solution[b] = temp;
}

// *** Add-Remove ***
#[derive(Clone)]
pub struct AddRem<X> {
    items: Vec<X>,
}

impl<X> AddRem<X> {
    pub fn new(items: Vec<X>) -> AddRem<X> {
        AddRem { items }
    }
}

impl<X: Clone> NeighbourGenerator<Vec<X>> for AddRem<X> {
    fn apply(&mut self, solution: &Solution<Vec<X>>) -> Solution<Vec<X>> {
        let mut solution = solution.clone();

        let mut rng = rand::thread_rng();
        let rn = rng.gen_range(0.0..2.0);
        let uniform_weights = vec![1; self.items.len()];

        if rn < 1.0 {
            // Add item
            add_item(&mut solution, &self.items, uniform_weights);
        } else if rn < 2.0 {
            // Remove item
            remove_item(&mut solution);
        }

        solution
    }

    fn update(&mut self, _: f64, _: f64) {
        // Do nothing
    }
}

// *** Swap ***
#[derive(Clone)]
pub struct Swap {}

impl Swap {
    pub fn new() -> Swap {
        Swap {}
    }
}

impl<X: Clone> NeighbourGenerator<Vec<X>> for Swap {
    fn apply(&mut self, solution: &Solution<Vec<X>>) -> Solution<Vec<X>> {
        let mut solution = solution.clone();
        swap_servers(&mut solution);

        solution
    }

    fn update(&mut self, _: f64, _: f64) {
        // Do nothing
    }
}

// *** Move ***
#[derive(Clone)]
pub struct Move {}

impl Move {
    pub fn new() -> Move {
        Move {}
    }
}

impl<X: Clone> NeighbourGenerator<Vec<X>> for Move {
    fn apply(&mut self, solution: &Solution<Vec<X>>) -> Solution<Vec<X>> {
        let mut solution = solution.clone();
        move_item(&mut solution);

        solution
    }

    fn update(&mut self, _: f64, _: f64) {
        // Do nothing
    }
}

// *** Add-Remove-Swap ***
#[derive(Clone)]
pub struct AddRemSwap<X> {
    items: Vec<X>,
}

impl<X> AddRemSwap<X> {
    pub fn new(items: Vec<X>) -> AddRemSwap<X> {
        AddRemSwap { items }
    }
}

impl<X: Clone> NeighbourGenerator<Vec<X>> for AddRemSwap<X> {
    fn apply(&mut self, solution: &Solution<Vec<X>>) -> Solution<Vec<X>> {
        let mut solution = solution.clone();

        let mut rng = rand::thread_rng();
        let rn = rng.gen_range(0.0..2.0);
        let uniform_weights = vec![1; self.items.len()];

        if rn < 1.0 {
            // Add item
            add_item(&mut solution, &self.items, uniform_weights);
        } else if rn < 2.0 {
            remove_item(&mut solution);
        } else {
            swap_servers(&mut solution);
        }

        solution
    }

    fn update(&mut self, _: f64, _: f64) {
        // Do nothing
    }
}

// *** Add-Remove-Move ***
#[derive(Clone)]
pub struct AddRemMove<X> {
    items: Vec<X>,
}

impl<X> AddRemMove<X> {
    pub fn new(items: Vec<X>) -> AddRemMove<X> {
        AddRemMove { items }
    }
}

impl<X: Clone> NeighbourGenerator<Vec<X>> for AddRemMove<X> {
    fn apply(&mut self, solution: &Solution<Vec<X>>) -> Solution<Vec<X>> {
        let mut solution = solution.clone();

        let mut rng = rand::thread_rng();
        let rn = rng.gen_range(0.0..2.0);
        let uniform_weights = vec![1; self.items.len()];

        if rn < 1.0 {
            // Add item
            add_item(&mut solution, &self.items, uniform_weights);
        } else if rn < 2.0 {
            remove_item(&mut solution);
        } else {
            move_item(&mut solution);
        }

        solution
    }

    fn update(&mut self, _: f64, _: f64) {
        // Do nothing
    }
}

// *** Data driven ***
#[derive(Clone)]
pub struct LastNumTracker {
    total_actions: usize,
    max_stored_actions: usize,
    last_actions: Vec<usize>,
    good_actions: Vec<bool>,
    action_weights: Vec<usize>,
}

impl LastNumTracker {
    pub fn new(max_stored_actions: usize) -> LastNumTracker {
        LastNumTracker {
            total_actions: 0,
            max_stored_actions,
            last_actions: Vec::new(),
            good_actions: Vec::new(),
            action_weights: vec![1, 1, 1],
        }
    }

    pub fn update_tracker(&mut self, last_action: usize, was_good: bool) {
        if self.good_actions.len() < self.max_stored_actions {
            self.good_actions.push(was_good);
            self.last_actions.push(last_action);

            if was_good {
                self.action_weights[last_action] += 1;
            }
        } else {
            let pos = self.total_actions % self.max_stored_actions;

            // Remove old action
            if self.good_actions[pos] {
                self.action_weights[self.last_actions[pos]] -= 1;
            }

            self.good_actions[pos] = was_good;
            self.last_actions[pos] = last_action;

            // Update new action
            if was_good {
                self.action_weights[last_action] += 1;
            }
        }

        self.total_actions += 1;
    }
}

// *** DD Last N: Add-Remove-Move ***
// *** Add-Remove-Move ***
#[derive(Clone)]
pub struct LastNumARM<X> {
    items: Vec<X>,
    last_action: usize,
    tracker: LastNumTracker,
}

impl<X> LastNumARM<X> {
    pub fn new(items: Vec<X>, max_stored_actions: usize) -> LastNumARM<X> {
        let tracker = LastNumTracker::new(max_stored_actions);
        LastNumARM {
            items,
            tracker,
            last_action: 0,
        }
    }
}

impl<X: Clone> NeighbourGenerator<Vec<X>> for LastNumARM<X> {
    fn apply(&mut self, solution: &Solution<Vec<X>>) -> Solution<Vec<X>> {
        let mut solution = solution.clone();

        let mut rng = rand::thread_rng();

        let dist = WeightedIndex::new(&self.tracker.action_weights).unwrap();
        self.last_action = dist.sample(&mut rng);

        match self.last_action {
            0 => add_item(&mut solution, &self.items, vec![1; self.items.len()]),
            1 => remove_item(&mut solution),
            2 => move_item(&mut solution),
            _ => panic!("Unexpected action value"),
        };

        solution
    }

    fn update(&mut self, obj_before: f64, obj_after: f64) {
        let was_good = obj_after > obj_before;
        self.tracker.update_tracker(self.last_action, was_good);
    }
}

// *** DD Last N: Add-Remove-Swap ***
// *** Add-Remove-Swap ***
#[derive(Clone)]
pub struct LastNumARS<X> {
    items: Vec<X>,
    last_action: usize,
    tracker: LastNumTracker,
}

impl<X> LastNumARS<X> {
    pub fn new(items: Vec<X>, max_stored_actions: usize) -> LastNumARM<X> {
        let tracker = LastNumTracker::new(max_stored_actions);
        LastNumARM {
            items,
            tracker,
            last_action: 0,
        }
    }
}

impl<X: Clone> NeighbourGenerator<Vec<X>> for LastNumARS<X> {
    fn apply(&mut self, solution: &Solution<Vec<X>>) -> Solution<Vec<X>> {
        let mut solution = solution.clone();

        let mut rng = rand::thread_rng();
        let dist = WeightedIndex::new(&self.tracker.action_weights).unwrap();
        self.last_action = dist.sample(&mut rng);

        match self.last_action {
            0 => add_item(&mut solution, &self.items, vec![1; self.items.len()]),
            1 => remove_item(&mut solution),
            2 => swap_servers(&mut solution),
            _ => panic!("Unexpected action value"),
        };

        solution
    }

    fn update(&mut self, obj_before: f64, obj_after: f64) {
        let was_good = obj_after > obj_before;
        self.tracker.update_tracker(self.last_action, was_good);
    }
}

// *** Thompson Sampling ***
#[derive(Clone)]
pub struct ThompsonSampling<X> {
    items: Vec<X>,
    last_action: usize,
    alphas: Vec<f64>,
    betas: Vec<f64>,
    c: f64,
}

impl<X> ThompsonSampling<X> {
    pub fn new(items: Vec<X>, c: f64) -> ThompsonSampling<X> {
        ThompsonSampling {
            items,
            last_action: 0,
            alphas: vec![2.0; 3],
            betas: vec![2.0; 3],
            c,
        }
    }
}

impl<X: Clone> NeighbourGenerator<Vec<X>> for ThompsonSampling<X> {
    fn apply(&mut self, solution: &Solution<Vec<X>>) -> Solution<Vec<X>> {
        let mut solution = solution.clone();
        let mut rng = rand::thread_rng();

        let (action, _) = (0..3)
            .into_iter()
            .map(|i| {
                (
                    i,
                    Beta::new(self.alphas[i], self.betas[i])
                        .unwrap()
                        .sample(&mut rng),
                )
            })
            .max_by(|(_, a_v), (_, b_v)| a_v.partial_cmp(&b_v).unwrap())
            .unwrap();

        self.last_action = action;

        match self.last_action {
            0 => add_item(&mut solution, &self.items, vec![1; self.items.len()]),
            1 => remove_item(&mut solution),
            2 => swap_servers(&mut solution),
            _ => panic!("Unexpected action value"),
        };

        solution
    }

    fn update(&mut self, obj_before: f64, obj_after: f64) {
        let reward = if obj_after > obj_before { 1.0 } else { 0.0 };

        let n = self.last_action;
        let c = self.c;

        if self.alphas[n] + self.betas[n] < self.c {
            self.alphas[n] += reward;
            self.betas[n] += 1.0 - reward;
        } else {
            self.alphas[n] += reward * (c / c + 1.0);
            self.betas[n] += (1.0 - reward) * (c / c + 1.0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_rem_swap_mutation() {
        let base_solution = Solution::new(vec![
            vec![0, 1],
            vec![],
            vec![3],
            vec![2, 4, 5],
            vec![],
            vec![],
            vec![],
            vec![6],
            vec![],
            vec![],
            vec![1],
        ]);

        let mut ngen = AddRemSwap::new(vec![0, 1, 2, 3, 4, 5, 6]);
        for _ in 0..1000 {
            let new_solution = ngen.apply(&base_solution);

            let (total_vms, servers_changed) =
                count_changes(&base_solution.point, &new_solution.point);

            assert!(total_vms == 7 || total_vms == 8 || total_vms == 9);
            assert!(servers_changed <= 2);
        }
    }

    #[test]
    fn test_add_rem_mutation() {
        let base_solution = Solution::new(vec![
            vec![0, 1],
            vec![],
            vec![3],
            vec![2, 4, 5],
            vec![],
            vec![],
            vec![],
            vec![6],
            vec![],
            vec![],
            vec![1],
        ]);

        let mut ngen = AddRem::new(vec![0, 1, 2, 3, 4, 5, 6]);
        for _ in 0..1000 {
            let new_solution = ngen.apply(&base_solution);

            let (total_vms, servers_changed) =
                count_changes(&base_solution.point, &new_solution.point);

            assert!(total_vms == 7 || total_vms == 8 || total_vms == 9);
            assert_eq!(servers_changed, 1);
        }
    }

    #[test]
    fn test_move_mutation() {
        let base_solution = Solution::new(vec![
            vec![0, 1],
            vec![],
            vec![3],
            vec![2, 4, 5],
            vec![],
            vec![],
            vec![],
            vec![6],
            vec![],
            vec![],
            vec![1],
        ]);

        let mut ngen = Move::new();
        for _ in 0..1000 {
            let new_solution = ngen.apply(&base_solution);

            let (total_vms, servers_changed) =
                count_changes(&base_solution.point, &new_solution.point);

            assert_eq!(total_vms, 8);
            assert_eq!(servers_changed, 2);
        }
    }

    #[test]
    fn test_dd_arm() {
        let base_solution = Solution::new(vec![
            vec![0, 1],
            vec![],
            vec![3],
            vec![2, 4, 5],
            vec![],
            vec![],
            vec![],
            vec![6],
            vec![],
            vec![],
            vec![1],
        ]);

        // --- Default case, all mutations balanced
        let mut def_num_increasing = 0;
        let mut def_num_swapped = 0;
        let mut def_num_decreasing = 0;

        let mut ngen = LastNumARM::new(vec![0, 1, 2, 3, 4, 5, 6], 100);
        for _ in 0..10000 {
            let new_solution = ngen.apply(&base_solution);
            let (num_increasing, num_swapped, num_decreasing) =
                count_differences(&base_solution.point, &new_solution.point);

            def_num_increasing += num_increasing;
            def_num_swapped += num_swapped;
            def_num_decreasing += num_decreasing;

            // No rewards
            ngen.update(0.0, 0.0);
        }

        // --- Increase additions
        let mut incr_num_increasing = 0;
        let mut incr_num_swapped = 0;
        let mut incr_num_decreasing = 0;

        let mut ngen = LastNumARM::new(vec![0, 1, 2, 3, 4, 5, 6], 100);
        for _ in 0..10000 {
            let new_solution = ngen.apply(&base_solution);
            let (num_increasing, num_swapped, num_decreasing) =
                count_differences(&base_solution.point, &new_solution.point);

            incr_num_increasing += num_increasing;
            incr_num_swapped += num_swapped;
            incr_num_decreasing += num_decreasing;

            // Reward increasing moves
            if num_increasing > 0 {
                ngen.update(0.0, 10.0);
            } else {
                ngen.update(0.0, 0.0);
            }
        }

        // --- Increase removals
        let mut decr_num_increasing = 0;
        let mut decr_num_swapped = 0;
        let mut decr_num_decreasing = 0;

        let mut ngen = LastNumARM::new(vec![0, 1, 2, 3, 4, 5, 6], 100);
        for _ in 0..10000 {
            let new_solution = ngen.apply(&base_solution);
            let (num_increasing, num_swapped, num_decreasing) =
                count_differences(&base_solution.point, &new_solution.point);

            decr_num_increasing += num_increasing;
            decr_num_swapped += num_swapped;
            decr_num_decreasing += num_decreasing;

            // Reward increasing moves
            if num_decreasing > 0 {
                ngen.update(0.0, 10.0);
            } else {
                ngen.update(0.0, 0.0);
            }
        }

        // --- Increase swaps
        let mut swap_num_increasing = 0;
        let mut swap_num_swapped = 0;
        let mut swap_num_decreasing = 0;

        let mut ngen = LastNumARM::new(vec![0, 1, 2, 3, 4, 5, 6], 100);
        for _ in 0..10000 {
            let new_solution = ngen.apply(&base_solution);
            let (num_increasing, num_swapped, num_decreasing) =
                count_differences(&base_solution.point, &new_solution.point);

            swap_num_increasing += num_increasing;
            swap_num_swapped += num_swapped;
            swap_num_decreasing += num_decreasing;

            // Reward increasing moves
            if num_swapped > 0 {
                ngen.update(0.0, 10.0);
            } else {
                ngen.update(0.0, 0.0);
            }
        }

        // --- Addition then swaps
        let mut chng_num_increasing = 0;
        let mut chng_num_swapped = 0;
        let mut chng_num_decreasing = 0;

        let mut ngen = LastNumARM::new(vec![0, 1, 2, 3, 4, 5, 6], 100);
        for i in 0..10000 {
            let new_solution = ngen.apply(&base_solution);
            let (num_increasing, num_swapped, num_decreasing) =
                count_differences(&base_solution.point, &new_solution.point);

            chng_num_increasing += num_increasing;
            chng_num_swapped += num_swapped;
            chng_num_decreasing += num_decreasing;

            // Reward increasing moves
            if num_increasing > 0 && i < 5000 || num_swapped > 0 && i >= 5000 {
                ngen.update(0.0, 10.0);
            } else {
                ngen.update(0.0, 0.0);
            }
        }

        assert!(def_num_increasing < 5000);
        assert!(def_num_swapped < 5000);
        assert!(def_num_decreasing < 5000);
        assert_eq!(
            def_num_increasing + def_num_swapped + def_num_decreasing,
            10000
        );

        assert!(incr_num_increasing > 9000);
        assert!(incr_num_swapped < 500);
        assert!(incr_num_decreasing < 500);
        assert_eq!(
            incr_num_increasing + incr_num_swapped + incr_num_decreasing,
            10000
        );

        assert!(decr_num_increasing < 500);
        assert!(decr_num_swapped < 500);
        assert!(decr_num_decreasing > 9000);
        assert_eq!(
            decr_num_increasing + decr_num_swapped + decr_num_decreasing,
            10000
        );

        assert!(swap_num_increasing < 500);
        assert!(swap_num_swapped > 9000);
        assert!(swap_num_decreasing < 500);
        assert_eq!(
            swap_num_increasing + swap_num_swapped + swap_num_decreasing,
            10000
        );

        assert!(chng_num_increasing > 4500);
        assert!(chng_num_swapped > 4500);
        assert!(chng_num_decreasing < 500);
        assert_eq!(
            chng_num_increasing + chng_num_swapped + chng_num_decreasing,
            10000
        );
    }

    #[test]
    fn test_thompson_sampling() {
        let base_solution = Solution::new(vec![
            vec![0, 1],
            vec![],
            vec![3],
            vec![2, 4, 5],
            vec![],
            vec![],
            vec![],
            vec![6],
            vec![],
            vec![],
            vec![1],
        ]);

        // --- Default case, all mutations balanced
        let mut def_num_increasing = 0;
        let mut def_num_swapped = 0;
        let mut def_num_decreasing = 0;

        let mut ngen = ThompsonSampling::new(vec![0, 1, 2, 3, 4, 5, 6], 5.0);
        for _ in 0..10000 {
            let new_solution = ngen.apply(&base_solution);
            let (num_increasing, num_swapped, num_decreasing) =
                count_differences(&base_solution.point, &new_solution.point);

            def_num_increasing += num_increasing;
            def_num_swapped += num_swapped;
            def_num_decreasing += num_decreasing;

            // No rewards
            ngen.update(0.0, 0.0);
        }

        // --- Increase additions
        let mut incr_num_increasing = 0;
        let mut incr_num_swapped = 0;
        let mut incr_num_decreasing = 0;

        let mut ngen = ThompsonSampling::new(vec![0, 1, 2, 3, 4, 5, 6], 5.0);
        for _ in 0..10000 {
            let new_solution = ngen.apply(&base_solution);
            let (num_increasing, num_swapped, num_decreasing) =
                count_differences(&base_solution.point, &new_solution.point);

            incr_num_increasing += num_increasing;
            incr_num_swapped += num_swapped;
            incr_num_decreasing += num_decreasing;

            // Reward increasing moves
            if num_increasing > 0 {
                ngen.update(0.0, 10.0);
            } else {
                ngen.update(0.0, 0.0);
            }
        }

        // --- Increase removals
        let mut decr_num_increasing = 0;
        let mut decr_num_swapped = 0;
        let mut decr_num_decreasing = 0;

        let mut ngen = ThompsonSampling::new(vec![0, 1, 2, 3, 4, 5, 6], 5.0);
        for _ in 0..10000 {
            let new_solution = ngen.apply(&base_solution);
            let (num_increasing, num_swapped, num_decreasing) =
                count_differences(&base_solution.point, &new_solution.point);

            decr_num_increasing += num_increasing;
            decr_num_swapped += num_swapped;
            decr_num_decreasing += num_decreasing;

            // Reward increasing moves
            if num_decreasing > 0 {
                ngen.update(0.0, 10.0);
            } else {
                ngen.update(0.0, 0.0);
            }
        }

        // --- Increase swaps
        let mut swap_num_increasing = 0;
        let mut swap_num_swapped = 0;
        let mut swap_num_decreasing = 0;

        let mut ngen = ThompsonSampling::new(vec![0, 1, 2, 3, 4, 5, 6], 5.0);
        for _ in 0..10000 {
            let new_solution = ngen.apply(&base_solution);
            let (num_increasing, num_swapped, num_decreasing) =
                count_differences(&base_solution.point, &new_solution.point);

            swap_num_increasing += num_increasing;
            swap_num_swapped += num_swapped;
            swap_num_decreasing += num_decreasing;

            // Reward increasing moves
            if num_swapped > 0 {
                ngen.update(0.0, 10.0);
            } else {
                ngen.update(0.0, 0.0);
            }
        }

        // --- Addition then swaps
        let mut chng_num_increasing = 0;
        let mut chng_num_swapped = 0;
        let mut chng_num_decreasing = 0;

        let mut ngen = ThompsonSampling::new(vec![0, 1, 2, 3, 4, 5, 6], 5.0);
        for i in 0..10000 {
            let new_solution = ngen.apply(&base_solution);
            let (num_increasing, num_swapped, num_decreasing) =
                count_differences(&base_solution.point, &new_solution.point);

            chng_num_increasing += num_increasing;
            chng_num_swapped += num_swapped;
            chng_num_decreasing += num_decreasing;

            // Reward increasing moves
            if num_increasing > 0 && i < 5000 || num_swapped > 0 && i >= 5000 {
                ngen.update(0.0, 10.0);
            } else {
                ngen.update(0.0, 0.0);
            }
        }

        assert!(def_num_increasing < 5000);
        assert!(def_num_swapped < 5000);
        assert!(def_num_decreasing < 5000);
        assert_eq!(
            def_num_increasing + def_num_swapped + def_num_decreasing,
            10000
        );

        assert!(incr_num_increasing > 9000);
        assert!(incr_num_swapped < 500);
        assert!(incr_num_decreasing < 500);
        assert_eq!(
            incr_num_increasing + incr_num_swapped + incr_num_decreasing,
            10000
        );

        assert!(decr_num_increasing < 500);
        assert!(decr_num_swapped < 500);
        assert!(decr_num_decreasing > 9000);
        assert_eq!(
            decr_num_increasing + decr_num_swapped + decr_num_decreasing,
            10000
        );

        assert!(swap_num_increasing < 500);
        assert!(swap_num_swapped > 9000);
        assert!(swap_num_decreasing < 500);
        assert_eq!(
            swap_num_increasing + swap_num_swapped + swap_num_decreasing,
            10000
        );

        assert!(chng_num_increasing > 4000);
        assert!(chng_num_swapped > 4000);
        assert!(chng_num_decreasing < 500);
        assert_eq!(
            chng_num_increasing + chng_num_swapped + chng_num_decreasing,
            10000
        );
    }

    fn count_changes<X>(base_servers: &Vec<Vec<X>>, new_servers: &Vec<Vec<X>>) -> (usize, usize) {
        let mut total_vms = 0;
        let mut servers_changed = 0;

        for i in 0..new_servers.len() {
            total_vms += new_servers[i].len();

            if new_servers[i].len() != base_servers[i].len() {
                servers_changed += 1;
            }
        }

        (total_vms, servers_changed)
    }

    fn count_differences<X>(
        base_servers: &Vec<Vec<X>>,
        new_servers: &Vec<Vec<X>>,
    ) -> (usize, usize, usize) {
        let mut num_increasing = 0;
        let mut num_decreaseing = 0;
        let mut num_swapped = 0;

        // Count current VMs
        let prev_vms = count_vms(&base_servers);

        // Count new VMs
        let new_vms = count_vms(&new_servers);

        if prev_vms > new_vms {
            num_increasing = num_increasing + 1;
        } else if prev_vms == new_vms {
            num_swapped = num_swapped + 1;
        } else if prev_vms < new_vms {
            num_decreaseing = num_decreaseing + 1;
        }

        (num_increasing, num_swapped, num_decreaseing)
    }

    fn count_vms<X>(servers: &Vec<Vec<X>>) -> usize {
        let mut total_vms = 0;

        for i in 0..servers.len() {
            total_vms += servers[i].len();
        }

        total_vms
    }
}
