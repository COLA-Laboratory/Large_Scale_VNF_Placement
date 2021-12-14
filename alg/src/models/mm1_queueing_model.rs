use crate::models::datacentre::Datacentre;
use crate::models::iterative_queueing_model::set_all_arrival_rates;
use crate::models::service::{Service, ServiceID};
use crate::models::{calc_ma, get_metrics, iterate_route, Server};
use crate::operators::mapping::{NodeType, RouteNode};

#[derive(Clone)]
pub struct MM1QueueingModel<'a> {
    dc: &'a Datacentre,
    sw_sr: f64,
    active_cost: f64,
    idle_cost: f64,
}

impl<'a> MM1QueueingModel<'a> {
    pub fn new(dc: &Datacentre, sw_sr: f64, active_cost: f64, idle_cost: f64) -> MM1QueueingModel {
        MM1QueueingModel {
            dc,
            sw_sr,
            active_cost,
            idle_cost,
        }
    }

    pub fn evaluate(
        &self,
        services: &Vec<Service>,
        routes: &Vec<(ServiceID, Vec<RouteNode>)>,
    ) -> (Vec<f64>, f64) {
        let mut servers_mean: Vec<Server> = vec![Server::new(); self.dc.num_servers];
        let mut sw_arr_mean = vec![0.0; self.dc.num_components()];

        // Reset the entries and warm up the cache
        for i in 0..self.dc.num_components() {
            sw_arr_mean[i] = 0.0;
        }

        for i in 0..self.dc.num_servers {
            servers_mean[i].clear();
        }

        // Calculate arrival rate
        let sw_pl = vec![0.0; self.dc.num_components()];
        set_all_arrival_rates(
            &routes,
            &services,
            &mut sw_arr_mean,
            &sw_pl,
            &mut servers_mean,
        );

        // Calculate service latency
        let mut service_latency = vec![0.0; services.len()];

        let mut s_count = vec![0; services.len()];

        for (s_id, route) in routes {
            let mut node_pv = vec![0.0; route.len()]; // Probability of visiting this node
            node_pv[0] = 1.0;

            iterate_route(route, |curr| {
                let num_next = route[curr].next_nodes.len();

                for node in &route[curr].next_nodes {
                    node_pv[*node] += node_pv[curr] / num_next as f64;
                }
            });

            let mut latency = 0.0;
            for i in 1..route.len() {
                let rn = &route[i];
                let (arr, _) =
                    get_metrics(rn, *s_id, &sw_arr_mean, &sw_pl, &mut servers_mean).unwrap();

                let srv = match rn.node_type {
                    NodeType::Component(_) => self.sw_sr,
                    NodeType::VNF(_, stage) => {
                        let vnf = &services[*s_id].vnfs[stage];
                        vnf.service_rate
                    }
                };

                latency = latency + (calc_wt(arr, srv) * node_pv[i]);
            }

            service_latency[*s_id] = calc_ma(service_latency[*s_id], latency, s_count[*s_id]).0;
            s_count[*s_id] += 1;
        }

        // Calculate energy consumption
        let energy = self.get_energy_consumption(services, &servers_mean, &sw_arr_mean, self.sw_sr);

        (service_latency, energy)
    }

    fn get_energy_consumption(
        &self,
        services: &Vec<Service>,
        servers_mean: &Vec<Server>,
        sw_arr_mean: &Vec<f64>,
        sw_sr: f64,
    ) -> f64 {
        let mut sum_energy = 0.0;

        for i in 0..self.dc.num_components() {
            let utilisation;
            if self.dc.is_server(i) {
                let server_busy = calc_busy(sw_arr_mean[i], sw_sr);
                let mut p_none_busy = 1.0;

                for (&(s_id, pos), vnf) in &servers_mean[i] {
                    // Producing VNFs don't go towards energy consumption
                    if pos == 0 {
                        continue;
                    }

                    let vnf_info = &services[s_id].vnfs[pos];

                    let vm_not_busy = 1.0 - calc_busy(vnf.arrival_rate, vnf_info.service_rate);
                    p_none_busy = p_none_busy * vm_not_busy;
                }

                utilisation = 1.0 - ((1.0 - server_busy) * p_none_busy)
            } else {
                utilisation = calc_busy(sw_arr_mean[i], self.sw_sr)
            };

            if utilisation == 0.0 {
                continue;
            }

            sum_energy += (self.active_cost * utilisation) + (self.idle_cost * (1.0 - utilisation));
        }

        sum_energy
    }
}

fn calc_busy(arr_rate: f64, service_rate: f64) -> f64 {
    if arr_rate >= service_rate {
        1.0
    } else {
        arr_rate / service_rate
    }
}

fn calc_wt(arr_rate: f64, service_rate: f64) -> f64 {
    if arr_rate >= service_rate {
        std::f64::INFINITY
    } else {
        1.0 / (service_rate - arr_rate)
    }
}
