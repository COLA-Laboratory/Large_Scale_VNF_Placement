pub mod datacentre;
pub mod perc_len_model;
pub mod iterative_queueing_model;
pub mod mm1_queueing_model;
pub mod routing;
pub mod service;
pub mod utilisation_model;
pub mod constant_model;
pub mod utilisation_latency_model;

use std::collections::{BTreeMap, VecDeque};

use crate::operators::mapping::{NodeType, RouteNode};

// (ServiceID, Step) -> VnfMetrics
pub type Server = BTreeMap<(usize, usize), VnfMetrics>;

#[derive(Debug, Clone)]
pub struct VnfMetrics {
    pub arrival_rate: f64,
    pub packet_losses: f64,
}

pub fn iterate_route(route: &Vec<RouteNode>, mut apply: impl FnMut(usize)) {
    let mut num_routes: Vec<u32> = route.iter().map(|x| x.route_count).collect();
    let mut queue = VecDeque::new();
    queue.push_back(0);

    while let Some(curr) = queue.pop_front() {
        num_routes[curr] = num_routes[curr] - 1;

        if num_routes[curr] == 0 {
            apply(curr);

            for n in &route[curr].next_nodes {
                queue.push_back(*n);
            }
        }
    }
}

fn calc_ma(current_mean: f64, new_value: f64, num_points: usize) -> (f64, f64) {
    let new = current_mean + (new_value - current_mean) / (num_points + 1) as f64;
    (new, (new - current_mean).abs())
}

pub fn get_metrics(
    rn: &RouteNode,
    service_id: usize,
    sw_arr: &Vec<f64>,
    sw_pl: &Vec<f64>,
    servers: &Vec<Server>,
) -> Option<(f64, f64)> {
    match rn.node_type {
        NodeType::Component(dc_id) => Some((sw_arr[dc_id], sw_pl[dc_id])),
        NodeType::VNF(dc_id, stage) => {
            let vnf = &servers[dc_id].get(&(service_id, stage));

            if let Some(vnf) = vnf {
                Some((vnf.arrival_rate, vnf.packet_losses))
            } else {
                None
            }
        }
    }
}

// ----- Unit tests ---- //
#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::routing::RoutingTable;
    use crate::models::service::{Service, VNF};
    use crate::operators::mapping::{find_routes, RouteNode};

    #[test]
    fn test_ma() {
        let numbers = [
            12.0, 0.2, 20.5, 5.4, 17.0, 0.0, 4.8, 230.2, 2.43, 7.12, 19.2,
        ];
        let mut ma = 0.0;
        for i in 0..numbers.len() {
            let act_mean = &numbers.iter().take(i + 1).sum::<f64>() / (i + 1) as f64;
            ma = calc_ma(ma, numbers[i], i).0;

            assert!((ma - act_mean).abs() < 0.001);
        }
    }

    #[test]
    fn test_iterate_route() {
        let (route, _, _, _) = get_simple_route();
        let expected = vec![0, 1, 2];

        iterate_route(&route, |curr| {
            assert_eq!(curr, expected[curr]);
        });

        let (route, _, _, _) = get_branching_route();
        let expected: Vec<usize> = (0..10).into_iter().collect();
        iterate_route(&route, |curr| {
            assert_eq!(curr, expected[curr]);
        });
    }

    fn get_service(length: usize) -> Service {
        let service = Service {
            id: 0,
            prod_rate: 10.0,
            vnfs: Vec::new(),
        };

        let mut vnfs = Vec::new();
        for _ in 0..length {
            let vnf = VNF {
                size: 100,
                queue_length: 20,
                service_rate: 10.0,
            };

            vnfs.push(vnf);
        }

        service
    }

    // (routes, num_components, num_servers, num_vnfs)
    fn get_simple_route() -> (Vec<RouteNode>, usize, usize, usize) {
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

        (route, 1, 1, 2)
    }

    // (routes, num_components, num_servers, num_vnfs)
    fn get_branching_route() -> (Vec<RouteNode>, usize, usize, usize) {
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

        (route, 8, 2, 2)
    }

    fn parse_sim_to_model(
        source: &str,
        routing_tables: &Vec<RoutingTable>,
    ) -> (Vec<Service>, Vec<(usize, Vec<RouteNode>)>) {
        let route_str = source.split(";");

        let mut services = Vec::new();
        let mut routes = Vec::new();

        for (s_id, service) in route_str.enumerate() {
            let sequence: Vec<usize> = service
                .split(",")
                .map(|id| id.parse::<usize>().unwrap())
                .collect();

            let service = get_service(sequence.len());
            services.push(service);

            let route = find_routes(sequence, routing_tables);

            routes.push((s_id, route));
        }

        (services, routes)
    }
}
