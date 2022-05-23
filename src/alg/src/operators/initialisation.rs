use rand::prelude::*;

use crate::{models::service::Service, operators::solution::Solution};

pub trait InitPop<X> {
    fn apply(&self, pop_size: usize) -> Vec<Solution<X>>;
}

pub struct UniformInitialisation<'a> {
    services: &'a Vec<Service>,
    solution_length: usize,
    p_vnf: f64,
}

impl UniformInitialisation<'_> {
    pub fn new(
        services: &Vec<Service>,
        solution_length: usize,
        p_vnf: f64,
    ) -> UniformInitialisation {
        UniformInitialisation {
            services,
            solution_length,
            p_vnf,
        }
    }
}

impl<'a> InitPop<Vec<&'a Service>> for UniformInitialisation<'a> {
    fn apply(&self, pop_size: usize) -> Vec<Solution<Vec<&'a Service>>> {
        let mut population = Vec::new();
        let mut rng = rand::thread_rng();

        for _ in 0..pop_size {
            let mut new_solution = vec![Vec::new(); self.solution_length];

            for i in 0..self.solution_length {
                while rng.gen_range(0.0..1.0) < self.p_vnf {
                    let rn = rng.gen_range(0..self.services.len());
                    new_solution[i].push(&self.services[rn]);
                }
            }

            population.push(Solution::new(new_solution));
        }

        population
    }
}

pub struct ServiceAwareInitialisation<'a> {
    services: &'a Vec<Service>,
    solution_length: usize,
}

impl ServiceAwareInitialisation<'_> {
    pub fn new(services: &Vec<Service>, solution_length: usize) -> ServiceAwareInitialisation {
        ServiceAwareInitialisation {
            services,
            solution_length,
        }
    }
}

impl<'a> InitPop<Vec<&'a Service>> for ServiceAwareInitialisation<'a> {
    fn apply(&self, pop_size: usize) -> Vec<Solution<Vec<&'a Service>>> {
        let mut population = Vec::new();
        let mut rng = rand::thread_rng();

        let mut min_size = 0;
        for service in self.services {
            for vnf in &service.vnfs {
                min_size = min_size + vnf.size;
            }
        }

        // Total capacity. 100 is the capacity of each server.
        let max_size = self.solution_length * 100;

        for i in 0..pop_size {
            let mut new_solution = vec![Vec::new(); self.solution_length];
            let prop = i as f64 / pop_size as f64;

            // Add all service instances
            let mut num_placed = 0;

            let used_size = prop * max_size as f64;

            let num_instances = (used_size / min_size as f64) as usize;
            let num_instances = num_instances.max(1); // At least one instance

            for service in self.services {
                for _ in 0..num_instances {
                    let pos = rng.gen_range(0..self.solution_length);
                    new_solution[pos].push(service);

                    num_placed = num_placed + 1;
                }
            }

            population.push(Solution::new(new_solution));
        }

        population
    }
}

pub struct ImprovedServiceAwareInitialisation<'a> {
    services: &'a Vec<Service>,
    solution_length: usize,
}

impl ImprovedServiceAwareInitialisation<'_> {
    pub fn new(
        services: &Vec<Service>,
        solution_length: usize,
    ) -> ImprovedServiceAwareInitialisation {
        ImprovedServiceAwareInitialisation {
            services,
            solution_length,
        }
    }
}

impl<'a> InitPop<Vec<&'a Service>> for ImprovedServiceAwareInitialisation<'a> {
    fn apply(&self, pop_size: usize) -> Vec<Solution<Vec<&'a Service>>> {
        let mut population = Vec::new();
        let mut rng = rand::thread_rng();

        let mut min_size = 0;
        for service in self.services {
            for vnf in &service.vnfs {
                min_size = min_size + vnf.size;
            }
        }

        // Total capacity. 100 is the capacity of each server.
        let max_size = self.solution_length * 100;

        let min_i = 1.0;
        let max_i = max_size as f64 / min_size as f64;

        let change = (max_i - min_i) / pop_size as f64;

        for i in 0..pop_size {
            let mut new_solution = vec![Vec::new(); self.solution_length];
            let num_instances = min_i + change * i as f64;

            // Add all service instances
            for service in self.services {
                let mut p_placed = num_instances;

                while p_placed > 0.0 {
                    let rand = rng.gen_range(0.0..1.0);

                    if rand < p_placed {
                        let pos = rng.gen_range(0..self.solution_length);
                        new_solution[pos].push(service);
                    }

                    p_placed -= 1.0;
                }
            }

            population.push(Solution::new(new_solution));
        }

        population
    }
}

#[cfg(test)]
mod tests {
    use crate::models::service::VNF;
    use crate::operators::initialisation::*;

    #[test]
    fn test_uniform_initialisation() {
        let service_a = Service {
            id: 0,
            prod_rate: 0.0,
            vnfs: [10, 10, 10, 30]
                .iter()
                .map(|size: &usize| VNF {
                    queue_length: 0,
                    service_rate: 0.0,
                    size: *size,
                })
                .collect(),
        };
        let service_b = Service {
            id: 1,
            prod_rate: 0.0,
            vnfs: [20, 42, 30, 5]
                .iter()
                .map(|size: &usize| VNF {
                    queue_length: 0,
                    service_rate: 0.0,
                    size: *size,
                })
                .collect(),
        };

        let service_c = Service {
            id: 2,
            prod_rate: 0.0,
            vnfs: [71, 30, 80, 26]
                .iter()
                .map(|size: &usize| VNF {
                    queue_length: 0,
                    service_rate: 0.0,
                    size: *size,
                })
                .collect(),
        };
        let services = vec![service_a, service_b, service_c];

        let length = 1000;
        let num_ind = 30;

        let init = UniformInitialisation::new(&services, length, 0.3);
        let population = init.apply(num_ind);

        assert_eq!(population.len(), num_ind);

        for ind in population {
            assert_eq!(ind.len(), length);

            let mut num_service_instances = vec![0; services.len()];

            for server in ind.point {
                for service_s in server {
                    let pos = services.iter().position(|serv| serv.id == service_s.id);
                    assert!(pos.is_some());

                    num_service_instances[pos.unwrap()] += 1;
                }
            }

            for i in 0..3 {
                assert!(num_service_instances[i] > 0);
            }

            let total: usize = num_service_instances.iter().sum();
            assert!(total > 200, total < 500);
        }        
    }

    #[test]
    fn test_initialisation() {
        let service_a = Service {
            id: 0,
            prod_rate: 0.0,
            vnfs: [10, 10, 10, 30]
                .iter()
                .map(|size: &usize| VNF {
                    queue_length: 0,
                    service_rate: 0.0,
                    size: *size,
                })
                .collect(),
        };
        let service_b = Service {
            id: 1,
            prod_rate: 0.0,
            vnfs: [20, 42, 30, 5]
                .iter()
                .map(|size: &usize| VNF {
                    queue_length: 0,
                    service_rate: 0.0,
                    size: *size,
                })
                .collect(),
        };

        let service_c = Service {
            id: 2,
            prod_rate: 0.0,
            vnfs: [71, 30, 80, 26]
                .iter()
                .map(|size: &usize| VNF {
                    queue_length: 0,
                    service_rate: 0.0,
                    size: *size,
                })
                .collect(),
        };
        let services = vec![service_a, service_b, service_c];
        let service_used = [60, 97, 207];

        let length = 1000;
        let num_ind = 30;

        let init = ServiceAwareInitialisation::new(&services, length);
        let population = init.apply(num_ind);

        assert_eq!(population.len(), num_ind);

        let mut totals = vec![];

        for ind in population {
            assert_eq!(ind.len(), length);

            let mut num_service_instances = vec![0; services.len()];

            for server in ind.point {
                for service_s in server {
                    let pos = services.iter().position(|serv| serv.id == service_s.id);
                    assert!(pos.is_some());

                    num_service_instances[pos.unwrap()] += 1;
                }
            }

            for i in 0..3 {
                assert!(num_service_instances[i] > 0);
            }

            let total: usize = num_service_instances.iter().sum();
            totals.push(total);

            let used_capacity: usize = num_service_instances
                .iter()
                .enumerate()
                .map(|(i, n)| (n * service_used[i]) as usize)
                .sum();

            assert!(used_capacity < length * 100);
        }

        totals.sort();
        let mut num_equal = 0;
        for i in 0..totals.len() - 1 {
            if totals[i] == totals[i + 1] {
                num_equal = num_equal + 1;
            }
        }

        assert!(num_equal < 3);
    }

    #[test]
    fn test_improved_initialisation() {
        let service_a = Service {
            id: 0,
            prod_rate: 0.0,
            vnfs: [10, 10, 10, 30]
                .iter()
                .map(|size: &usize| VNF {
                    queue_length: 0,
                    service_rate: 0.0,
                    size: *size,
                })
                .collect(),
        };
        let service_b = Service {
            id: 1,
            prod_rate: 0.0,
            vnfs: [20, 42, 30, 5]
                .iter()
                .map(|size: &usize| VNF {
                    queue_length: 0,
                    service_rate: 0.0,
                    size: *size,
                })
                .collect(),
        };

        let service_c = Service {
            id: 2,
            prod_rate: 0.0,
            vnfs: [71, 30, 80, 26]
                .iter()
                .map(|size: &usize| VNF {
                    queue_length: 0,
                    service_rate: 0.0,
                    size: *size,
                })
                .collect(),
        };
        let services = vec![service_a, service_b, service_c];
        let service_used = [60, 97, 207];

        let length = 1000;
        let num_ind = 30;

        let init = ImprovedServiceAwareInitialisation::new(&services, length);
        let population = init.apply(num_ind);

        assert_eq!(population.len(), num_ind);

        let mut totals = vec![];

        for ind in population {
            assert_eq!(ind.len(), length);

            let mut num_service_instances = vec![0; services.len()];

            for server in ind.point {
                for service_s in server {
                    let pos = services.iter().position(|serv| serv.id == service_s.id);
                    assert!(pos.is_some());

                    num_service_instances[pos.unwrap()] += 1;
                }
            }

            for i in 0..3 {
                assert!(num_service_instances[i] > 0);
            }

            let total: usize = num_service_instances.iter().sum();
            totals.push(total);

            let used_capacity: usize = num_service_instances
                .iter()
                .enumerate()
                .map(|(i, n)| (n * service_used[i]) as usize)
                .sum();

            assert!(used_capacity < length * 100);
        }

        totals.sort();
        let mut num_equal = 0;
        for i in 0..totals.len() - 1 {
            if totals[i] == totals[i + 1] {
                num_equal = num_equal + 1;
            }
        }

        assert!(num_equal < 3);
    }
}
