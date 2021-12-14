use super::{
    datacentre::Datacentre,
    iterate_route,
    iterative_queueing_model::IterativeQueueingModel,
    service::{Service, ServiceID},
};
use crate::operators::mapping::RouteNode;

#[derive(Clone)]
pub struct ConstantModel<'a> {
    dc: &'a Datacentre,
    services: &'a Vec<Service>,
    qm: IterativeQueueingModel<'a>,
    waiting_time: f64,
    packet_loss: f64,
}

impl<'a> ConstantModel<'a> {
    pub fn new(
        dc: &'a Datacentre,
        services: &'a Vec<Service>,
        qm: IterativeQueueingModel<'a>,
        waiting_time: f64,
        packet_loss: f64,
    ) -> ConstantModel<'a> {
        ConstantModel {
            dc,
            services,
            qm,
            waiting_time,
            packet_loss,
        }
    }

    pub fn evaluate(
        &self,
        // placements: &Solution<Vec<&Service>>,
        routes: &Vec<(ServiceID, Vec<RouteNode>)>,
    ) -> (Vec<f64>, Vec<f64>, f64) {
        // Constant waiting time / pl - specific numbers don't matter
        let mut latencies = vec![0.0; self.services.len()];
        let mut pls = vec![0.0; self.services.len()];

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

            // Calculate expected latency / packet loss
            let mut expected_latency = 0.0;
            let mut expected_pl = 0.0;
            for i in 0..route.len() {
                expected_latency += node_ev[i] * self.waiting_time;
                expected_pl += node_ev[i] * self.packet_loss;
            }

            latencies[*service_id] = expected_latency;
            pls[*service_id] = expected_pl;
        }

        let (_, _, energy) = self.qm.evaluate(self.services, routes);

        (latencies, pls, energy)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::datacentre::FatTree;
    use crate::models::service::VNF;
    use crate::operators::mapping::NodeType;

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
        let model = ConstantModel::new(&dc, &services, qm, 1.0, 0.01);

        let simple_route = vec![(0, get_simple_route())];
        let branching_route = vec![(0, get_branching_route())];
        let multiple_services = vec![(0, get_simple_route()), (1, get_branching_route())];

        let (latencies_simp, pls_simp, _) = model.evaluate(&simple_route);
        let (latencies_branch, pls_branch, _) = model.evaluate(&branching_route);
        let (latencies_mult, pls_mult, _) = model.evaluate(&multiple_services);

        assert_eq!(latencies_simp[0], 3.0);
        assert_eq!(pls_simp[0], 0.01 * 3.0);

        assert!(latencies_branch[0] > 5.65 && latencies_branch[0] < 5.67);
        assert!(pls_branch[0] > 0.0565 && pls_branch[0] < 0.0567);

        assert_eq!(latencies_mult[0], 3.0);
        assert_eq!(pls_mult[0], 0.01 * 3.0);
        assert!(latencies_mult[1] > 5.65 && latencies_mult[1] < 5.67);
        assert!(pls_mult[1] > 0.0565 && pls_mult[1] < 0.0567);
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
