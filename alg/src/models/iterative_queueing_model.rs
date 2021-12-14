use crate::models::datacentre::Datacentre;
use crate::models::service::{Service, ServiceID};
use crate::models::{calc_ma, get_metrics, iterate_route, Server, VnfMetrics};
use crate::operators::mapping::{NodeType, RouteNode};

#[derive(Clone)]
pub struct IterativeQueueingModel<'a> {
    dc: &'a Datacentre,
    sw_sr: f64,
    sw_ql: usize,
    pub target_acc: f64,
    pub converged_iterations: usize,
    active_cost: f64,
    idle_cost: f64,
}

impl<'a> IterativeQueueingModel<'a> {
    pub fn new(
        dc: &Datacentre,
        sw_sr: f64,
        sw_ql: usize,
        accuracy: f64,
        converged_iterations: usize,
        active_cost: f64,
        idle_cost: f64,
    ) -> IterativeQueueingModel {
        IterativeQueueingModel {
            dc,
            sw_sr,
            sw_ql,
            target_acc: accuracy,
            converged_iterations,
            active_cost,
            idle_cost,
        }
    }

    pub fn evaluate(
        &self,
        services: &Vec<Service>,
        routes: &Vec<(ServiceID, Vec<RouteNode>)>,
    ) -> (Vec<f64>, Vec<f64>, f64) {
        let mut servers_curr: Vec<Server> = vec![Server::new(); self.dc.num_servers];
        let mut servers_prev: Vec<Server> = vec![Server::new(); self.dc.num_servers];
        let mut servers_mid: Vec<Server> = vec![Server::new(); self.dc.num_servers];

        let mut sw_arr_curr = vec![0.0; self.dc.num_components()];
        let mut sw_arr_prev = vec![0.0; self.dc.num_components()];
        let mut sw_arr_mid = vec![0.0; self.dc.num_components()];

        let mut sw_pl = vec![0.0; self.dc.num_components()];

        // Reset the entries to warm up the cache
        for i in 0..self.dc.num_servers {
            servers_curr[i].clear();
            servers_prev[i].clear();
            servers_mid[i].clear();
        }

        for i in 0..self.dc.num_components() {
            sw_arr_curr[i] = 0.0;
            sw_arr_prev[i] = 0.0;
            sw_arr_mid[i] = 0.0;
            sw_pl[i] = 0.0;
        }

        // Calculate arrival rate
        let mut num_iterations = 0;
        let mut num_below = 0;
        let mut max_diff: f64;

        set_all_arrival_rates(
            &routes,
            &services,
            &mut sw_arr_curr,
            &sw_pl,
            &mut servers_curr,
        );

        while num_below < self.converged_iterations {
            // Calculate packet loss for all components
            set_all_pl(
                &services,
                &mut sw_pl, // Switch info
                &sw_arr_curr,
                self.sw_sr,
                self.sw_ql,
                &mut servers_curr,
            );

            for i in 0..sw_arr_curr.len() {
                sw_arr_prev[i] = sw_arr_curr[i];
            }
            for i in 0..servers_curr.len() {
                for (key, vnf) in &servers_curr[i] {
                    servers_prev[i].insert(*key, vnf.clone());                    
                }
            }

            // Add arrival rates
            set_all_arrival_rates(
                &routes,
                &services,
                &mut sw_arr_curr,
                &sw_pl,
                &mut servers_curr,
            );

            // Cumulative moving average of arrival rates
            max_diff = 0.0;

            for i in 0..sw_arr_mid.len() {
                let mid = (sw_arr_curr[i] + sw_arr_prev[i]) / 2.0;
                let diff = (sw_arr_mid[i] - mid).abs();

                sw_arr_mid[i] = mid;

                max_diff = max_diff.max(diff);
            }
            for i in 0..servers_mid.len() {
                for (&(s_id, pos), _) in &servers_curr[i] {
                    let server_curr = servers_curr[i].get(&(s_id, pos)).unwrap();
                    let server_prev = servers_prev[i].get(&(s_id, pos)).unwrap();
                    let mut server_mid = servers_mid[i].entry((s_id, pos)).or_insert(VnfMetrics {
                        arrival_rate: 0.0,
                        packet_losses: 0.0,
                    });

                    let mid = (server_curr.arrival_rate + server_prev.arrival_rate) / 2.0;
                    let diff = (server_mid.arrival_rate - mid).abs();

                    server_mid.arrival_rate = mid;

                    max_diff = max_diff.max(diff);
                }
            }

            if max_diff < self.target_acc {
                num_below = num_below + 1;
            } else {
                num_below = 0;
            }

            num_iterations = num_iterations + 1;
        }

        // Recalculate PL using average arrival rate
        set_all_pl(
            &services,
            &mut sw_pl, // Switch info
            &sw_arr_curr,
            self.sw_sr,
            self.sw_ql,
            &mut servers_curr,
        );

        // Calculate service latency + pl
        let mut service_latency = vec![0.0; services.len()];
        let mut service_pl = vec![0.0; services.len()];

        let mut s_count = vec![0; services.len()];

        for (s_id, route) in routes {
            let mut node_pk = vec![0.0; route.len()]; // Probability a packet survives to this node
            let mut node_pl = vec![0.0; route.len()]; // Packet loss at this node
            let mut node_pv = vec![0.0; route.len()]; // Probability of visiting this node
            node_pv[0] = 1.0;
            node_pk[0] = 1.0;

            iterate_route(route, |curr| {
                let (_, pl) =
                    get_metrics(&route[curr], *s_id, &sw_arr_curr, &sw_pl, &mut servers_curr)
                        .unwrap();

                node_pl[curr] = pl;
                node_pk[curr] = node_pk[curr] * (1.0 - node_pl[curr]);

                let num_next = route[curr].next_nodes.len();
                if num_next == 0 {
                    service_pl[*s_id] =
                        calc_ma(service_pl[*s_id], 1.0 - node_pk[curr], s_count[*s_id]).0;
                }

                for node in &route[curr].next_nodes {
                    node_pk[*node] += node_pk[curr] / num_next as f64;
                    node_pv[*node] += node_pv[curr] / num_next as f64;
                }
            });

            let mut latency = 0.0;
            for i in 1..route.len() {
                let rn = &route[i];
                let (arr, _) =
                    get_metrics(rn, *s_id, &sw_arr_curr, &sw_pl, &mut servers_curr).unwrap();

                let (srv, ql) = match rn.node_type {
                    NodeType::Component(_) => (self.sw_sr, self.sw_ql),
                    NodeType::VNF(_, stage) => {
                        let vnf = &services[*s_id].vnfs[stage];
                        (vnf.service_rate, vnf.queue_length)
                    }
                };

                latency = latency + (calc_wt(arr, srv, ql, node_pl[i]) * node_pv[i]);
            }

            service_latency[*s_id] = calc_ma(service_latency[*s_id], latency, s_count[*s_id]).0;
            s_count[*s_id] += 1;
        }

        // Calculate energy consumption
        let energy = self.get_energy_consumption(
            services,
            &servers_curr,
            &sw_arr_curr,
            self.sw_sr,
            self.sw_ql,
        );

        (service_latency, service_pl, energy)
    }

    pub fn get_energy_consumption(
        &self,
        services: &Vec<Service>,
        servers_mean: &Vec<Server>,
        sw_arr_mean: &Vec<f64>,
        sw_sr: f64,
        sw_ql: usize,
    ) -> f64 {
        let mut sum_energy = 0.0;

        for i in 0..self.dc.num_components() {
            let utilisation;
            if self.dc.is_server(i) {
                let server_busy = calc_busy(sw_arr_mean[i], sw_sr, sw_ql);
                let mut p_none_busy = 1.0;

                for (&(s_id, pos), vnf) in &servers_mean[i] {
                    // Producing VNFs don't go towards energy consumption
                    if pos == 0 {
                        continue;
                    }

                    let vnf_info = &services[s_id].vnfs[pos];

                    let vm_not_busy = 1.0
                        - calc_busy(
                            vnf.arrival_rate,
                            vnf_info.service_rate,
                            vnf_info.queue_length,
                        );
                    p_none_busy = p_none_busy * vm_not_busy;
                }

                utilisation = 1.0 - ((1.0 - server_busy) * p_none_busy)
            } else {
                utilisation = calc_busy(sw_arr_mean[i], self.sw_sr, self.sw_ql)
            };

            if utilisation == 0.0 {
                continue;
            }

            sum_energy += (self.active_cost * utilisation) + (self.idle_cost * (1.0 - utilisation));
        }

        sum_energy
    }
}

fn set_all_pl(
    services: &Vec<Service>,
    sw_pl: &mut Vec<f64>,
    sw_arr: &Vec<f64>,
    sw_srv_rate: f64,
    sw_queue_length: usize,
    servers: &mut Vec<Server>,
) {
    for i in 0..sw_pl.len() {
        sw_pl[i] = calc_pl(sw_arr[i], sw_srv_rate, sw_queue_length);
    }

    for i in 0..servers.len() {
        for (&(s_id, pos), vnf_info) in servers[i].iter_mut() {
            // First VNF can't drop packets as it is emitting them
            if pos == 0 {
                continue;
            }

            let vnf = &services[s_id].vnfs[pos];
            vnf_info.packet_losses =
                calc_pl(vnf_info.arrival_rate, vnf.service_rate, vnf.queue_length);
        }
    }
}

fn calc_pl(arrival_rate: f64, service_rate: f64, queue_length: usize) -> f64 {
    let queue_length = queue_length as f64;
    let rho = arrival_rate / service_rate;

    if rho == 1. {
        1. / (queue_length + 1.)
    } else {
        ((1. - rho) * rho.powf(queue_length)) / (1. - rho.powf(queue_length + 1.))
    }
}

fn calc_wt(arrival_rate: f64, service_rate: f64, queue_length: usize, packet_loss: f64) -> f64 {
    let queue_length = queue_length as f64;

    let rho = arrival_rate / service_rate;

    if arrival_rate == 0. {
        return 0.;
    }

    let num_in_system = if rho != 1.0 {
        let a = rho
            * (1.0 - (queue_length + 1.0) * rho.powf(queue_length)
                + queue_length * rho.powf(queue_length + 1.0));
        let b = (1.0 - rho) * (1.0 - rho.powf(queue_length + 1.0));

        a / b
    } else {
        queue_length / 2.0
    };

    let ar = arrival_rate * (1.0 - packet_loss);

    num_in_system / ar
}

fn calc_busy(arrival_rate: f64, service_rate: f64, queue_length: usize) -> f64 {
    if arrival_rate > 0.0 && service_rate == 0.0 {
        return std::f64::INFINITY;
    }

    let rho = arrival_rate / service_rate;
    let k = queue_length as f64;

    let p_empty = if arrival_rate != service_rate {
        (1.0 - rho) / (1.0 - rho.powf(k + 1.0))
    } else {
        1.0 / (k + 1.0)
    };

    1.0 - p_empty
}

pub fn set_arrival_rate<'a>(
    arrival_rate: f64,
    rn: &RouteNode,
    service_id: usize,
    sw_arr: &'a mut Vec<f64>,
    servers: &'a mut Vec<Server>,
) {
    match rn.node_type {
        NodeType::Component(dc_id) => sw_arr[dc_id] = arrival_rate,
        NodeType::VNF(dc_id, stage) => {
            servers[dc_id]
                .entry((service_id, stage))
                .or_insert(VnfMetrics {
                    arrival_rate: 0.0,
                    packet_losses: 0.0,
                })
                .arrival_rate = arrival_rate;
        }
    }
}

pub fn set_all_arrival_rates(
    solution: &Vec<(ServiceID, Vec<RouteNode>)>,
    services: &Vec<Service>,
    sw_arr: &mut Vec<f64>,
    sw_pl: &Vec<f64>,
    servers: &mut Vec<Server>,
) {
    // Reset memory
    for i in 0..sw_arr.len() {
        sw_arr[i] = 0.0;
    }
    for i in 0..servers.len() {
        for vnf in servers[i].values_mut() {
            vnf.arrival_rate = 0.0;
        }
    }

    let mut num_instances = vec![0; services.len()];
    for (s_id, _) in solution {
        num_instances[*s_id] += 1;
    }

    for (s_id, route) in solution {
        let mut arrs = vec![0.0; route.len()];
        arrs[0] = services[*s_id].prod_rate / num_instances[*s_id] as f64;

        iterate_route(route, |curr| {
            let cn = &route[curr];
            let metrics = get_metrics(cn, *s_id, sw_arr, sw_pl, servers);

            let mut arr = 0.0;
            let mut pl = 0.0;
            if let Some((a, p)) = metrics {
                arr = a;
                pl = p;
            }

            set_arrival_rate(arr + arrs[curr], &cn, *s_id, sw_arr, servers);

            let eff_out = arrs[curr] * (1.0 - pl);
            let distr_out = eff_out / cn.next_nodes.len() as f64;

            for n_id in &cn.next_nodes {
                arrs[*n_id] = arrs[*n_id] + distr_out;
            }
        });
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
    fn test_set_arrival_rates() {
        // Simple
        let (simple_route, num_components, num_servers, num_vnfs) = get_simple_route();
        let simple_service = get_service(num_vnfs);
        let mut arr = vec![0.0; num_components];
        let pl = vec![0.0; num_components];
        let mut servers = vec![Server::new(); num_servers];

        set_all_arrival_rates(
            &vec![(0, simple_route)],
            &vec![simple_service],
            &mut arr,
            &pl,
            &mut servers,
        );

        assert_eq!(servers[0][&(0, 0)].arrival_rate, 10.0);
        assert_eq!(arr[0], 10.0);
        assert_eq!(servers[0][&(0, 1)].arrival_rate, 10.0);

        // Branching
        let (branch_route, num_components, num_servers, num_vnfs) = get_branching_route();
        let branching_service = get_service(num_vnfs);
        let mut arr = vec![0.0; num_components];
        let pl = vec![0.0; num_components];
        let mut servers = vec![Server::new(); num_servers];

        set_all_arrival_rates(
            &vec![(0, branch_route)],
            &vec![branching_service],
            &mut arr,
            &pl,
            &mut servers,
        );

        assert_eq!(arr[0], 10.0); // Server 1
        assert_eq!(servers[0][&(0, 0)].arrival_rate, 10.0);
        assert_eq!(arr[1], 10.0); // Server 2
        assert_eq!(servers[1][&(0, 1)].arrival_rate, 10.0);
        assert_eq!(arr[2], 5.0);
        assert_eq!(arr[3], 5.0);
        assert_eq!(arr[4], 5.0 / 3.0);
        assert_eq!(arr[5], 5.0 / 3.0);
        assert_eq!(arr[6], 5.0 / 3.0);
        assert_eq!(arr[6], 5.0 / 3.0);

        // Combined
        let (simple_route, _, _, _) = get_simple_route();
        let (branching_route, num_components, num_servers, num_vnfs) = get_branching_route();

        let simp_service = get_service(num_vnfs);
        let brnc_service = get_service(num_vnfs);

        let mut arr = vec![0.0; num_components];
        let pl = vec![0.0; num_components];
        let mut servers = vec![Server::new(); num_servers];

        set_all_arrival_rates(
            &vec![(0, simple_route), (1, branching_route)],
            &vec![simp_service, brnc_service],
            &mut arr,
            &pl,
            &mut servers,
        );

        assert_eq!(arr[0], 20.0); // Server 1
        assert_eq!(servers[0][&(0, 0)].arrival_rate, 10.0);
        assert_eq!(servers[0][&(0, 1)].arrival_rate, 10.0);
        assert_eq!(servers[0][&(1, 0)].arrival_rate, 10.0);
        assert_eq!(arr[1], 10.0); // Server 2
        assert_eq!(servers[1][&(1, 1)].arrival_rate, 10.0);
        assert_eq!(arr[2], 5.0);
        assert_eq!(arr[3], 5.0);
        assert_eq!(arr[4], 5.0 / 3.0);
        assert_eq!(arr[5], 5.0 / 3.0);
        assert_eq!(arr[6], 5.0 / 3.0);
        assert_eq!(arr[6], 5.0 / 3.0);

        // Lossy Simple
        let (simple_route, num_components, num_servers, num_vnfs) = get_simple_route();
        let simple_service = get_service(num_vnfs);
        let mut arr = vec![0.0; num_components];
        let pl = vec![0.5; num_components];
        let mut servers = vec![Server::new(); num_servers];

        set_all_arrival_rates(
            &vec![(0, simple_route)],
            &vec![simple_service],
            &mut arr,
            &pl,
            &mut servers,
        );

        assert_eq!(servers[0][&(0, 0)].arrival_rate, 10.0);
        assert_eq!(arr[0], 10.0);
        assert_eq!(servers[0][&(0, 1)].arrival_rate, 5.0);
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
