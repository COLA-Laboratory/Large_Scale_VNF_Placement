use rand::{random, Rng};

use super::solution::Solution;

pub trait Crossover<X> {
    fn apply(&self, parent_one: &Solution<X>, parent_two: &Solution<X>) -> Vec<Solution<X>>;
}

pub struct UniformCrossover {
    pc: f64,
}

impl UniformCrossover {
    pub fn new(pc: f64) -> UniformCrossover {
        UniformCrossover { pc }
    }
}

impl<X: Clone> Crossover<Vec<X>> for UniformCrossover {
    fn apply(
        &self,
        parent_one: &Solution<Vec<X>>,
        parent_two: &Solution<Vec<X>>,
    ) -> Vec<Solution<Vec<X>>> {
        let mut rng = rand::thread_rng();

        if rng.gen::<f64>() > self.pc {
            return vec![parent_one.clone(), parent_two.clone()];
        }

        let length = parent_one.len();

        let mut child_a = Vec::with_capacity(length);
        let mut child_b = Vec::with_capacity(length);

        for i in 0..length {
            if random() {
                child_a.push(parent_one[i].clone());
                child_b.push(parent_two[i].clone());
            } else {
                child_a.push(parent_two[i].clone());
                child_b.push(parent_one[i].clone());
            }
        }

        vec![Solution::new(child_a), Solution::new(child_b)]
    }
}
