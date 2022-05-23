use super::{
    datacentre::Datacentre,
    iterate_route,
    iterative_queueing_model::IterativeQueueingModel,
    service::{Service, ServiceID},
};
use crate::operators::mapping::{NodeType, RouteNode};
use std::collections::HashSet;

#[derive(Clone)]
pub struct UtilisationLatencyModel<'a> {
    dc: &'a Datacentre,
    services: &'a Vec<Service>,
    qm: IterativeQueueingModel<'a>,
    capacities: Vec<usize>,
}

impl<'a> UtilisationLatencyModel<'a> {
    pub fn new(
        dc: &'a Datacentre,
        services: &'a Vec<Service>,
        qm: IterativeQueueingModel<'a>,
        capacities: Vec<usize>,
    ) -> UtilisationLatencyModel<'a> {
        UtilisationLatencyModel {
            dc,
            services,
            qm,
            capacities,
        }
    }

    pub fn evaluate(
        &self,
        // placements: &Solution<Vec<&Service>>,
        routes: &Vec<(ServiceID, Vec<RouteNode>)>,
    ) -> (Vec<f64>, f64) {
        // Constant waiting time / pl - specific numbers don't matter
        let mut latencies = vec![0.0; self.services.len()];

        // Sum capacity of distinct VNFs
        let mut counted_vnfs = HashSet::new();
        let mut server_full = vec![0; self.capacities.len()];

        for (service_id, route) in routes {
            // Calculate expected latency
            for node in route {
                if let NodeType::VNF(server_id, stage) = node.node_type {
                    if counted_vnfs.contains(&(server_id, service_id, stage)) {
                        continue;
                    }

                    counted_vnfs.insert((server_id, service_id, stage));
                    server_full[server_id] += self.services[*service_id].vnfs[stage].size;
                }
            }
        }

        for (service_id, route) in routes {
            let mut node_ev = vec![0.0; route.len()]; // Expected number of visits to this node
            node_ev[0] = 1.0;

            iterate_route(route, |curr| {
                // Probability of visiting each node
                let num_next = route[curr].next_nodes.len();
                for node in &route[curr].next_nodes {
                    node_ev[*node] += node_ev[curr] / num_next as f64;
                }
            });

            let mut latency = 0.0;
            for i in 1..route.len() {
                let node = &route[i];

                latency += node_ev[i]
                    * match node.node_type {
                        NodeType::Component(_) => 1.0,
                        NodeType::VNF(node_id, _) => {
                            server_full[node_id] as f64 / self.capacities[node_id] as f64
                        }
                    };
            }

            latencies[*service_id] = latency;
        }

        let (_, _, energy) = self.qm.evaluate(self.services, routes);

        (latencies, energy)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::datacentre::FatTree;
    use crate::models::service::VNF;

    #[test]
    pub fn test_evaluate() {
        let dc = FatTree::new(4);
        let qm = IterativeQueueingModel::new(&dc, 10.0, 10, 0.05, 10, 100.0, 30.0);

        let service_a = Service {
            id: 0,
            prod_rate: 5.0,
            vnfs: [50, 50]
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
            prod_rate: 5.0,
            vnfs: [40, 30]
                .iter()
                .map(|size: &usize| VNF {
                    queue_length: 0,
                    service_rate: 0.0,
                    size: *size,
                })
                .collect(),
        };
        let services = vec![service_a, service_b];
        let capacities = vec![100; 16];
        let model = UtilisationLatencyModel::new(&dc, &services, qm, capacities);

        let simple_route = vec![(0, get_simple_route())];
        let simple_route_two = vec![(1, get_simple_route())];
        let branching_route = vec![(0, get_branching_route())];
        let multiple_services = vec![(0, get_simple_route()), (1, get_branching_route())];

        let (latencies_simp, _) = model.evaluate(&simple_route);
        let (latencies_simp_two, _) = model.evaluate(&simple_route_two);
        let (latencies_branch, _) = model.evaluate(&branching_route);
        let (latencies_mult, _) = model.evaluate(&multiple_services);

        assert_eq!(latencies_simp[0], 2.0);
        assert_eq!(latencies_simp_two[1], 1.7);

        assert!(latencies_branch[0] > 4.15 && latencies_branch[0] < 4.17);

        assert_eq!(latencies_mult[0], 2.40);
        assert!(latencies_mult[1] > 3.95 && latencies_mult[1] < 3.97);
    }

    // (routes, num_components, num_servers, num_vnfs)
    fn get_simple_route() -> Vec<RouteNode> {
        let route = vec![
            RouteNode {
                node_type: NodeType::VNF(0, 0),
                route_count: 1,
                next_nodes: vec![1],
            },
            RouteNode {
                node_type: NodeType::Component(0),
                route_count: 1,
                next_nodes: vec![2],
            },
            RouteNode {
                node_type: NodeType::VNF(0, 1),
                route_count: 1,
                next_nodes: vec![],
            },
        ];

        route
    }

    // (routes, num_components, num_servers, num_vnfs)
    fn get_branching_route() -> Vec<RouteNode> {
        let route = vec![
            RouteNode {
                node_type: NodeType::VNF(0, 0),
                route_count: 1,
                next_nodes: vec![1],
            },
            RouteNode {
                node_type: NodeType::Component(0),
                route_count: 1,
                next_nodes: vec![2, 3],
            },
            RouteNode {
                node_type: NodeType::Component(2),
                route_count: 1,
                next_nodes: vec![4, 5, 8],
            },
            RouteNode {
                node_type: NodeType::Component(3),
                route_count: 1,
                next_nodes: vec![6, 7, 8],
            },
            RouteNode {
                node_type: NodeType::Component(4),
                route_count: 1,
                next_nodes: vec![8],
            },
            RouteNode {
                node_type: NodeType::Component(5),
                route_count: 1,
                next_nodes: vec![8],
            },
            RouteNode {
                node_type: NodeType::Component(6),
                route_count: 1,
                next_nodes: vec![8],
            },
            RouteNode {
                node_type: NodeType::Component(7),
                route_count: 1,
                next_nodes: vec![8],
            },
            RouteNode {
                node_type: NodeType::Component(1),
                route_count: 6,
                next_nodes: vec![9],
            },
            RouteNode {
                node_type: NodeType::VNF(1, 1),
                route_count: 1,
                next_nodes: vec![],
            },
        ];

        route
    }
}
