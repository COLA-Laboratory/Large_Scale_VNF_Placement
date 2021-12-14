use super::{
    datacentre::Datacentre,
    service::{Service, ServiceID},
};
use crate::operators::mapping::{NodeType, RouteNode};
use std::collections::HashSet;

#[derive(Clone)]
pub struct PercLenModel<'a> {
    dc: &'a Datacentre,
    services: &'a Vec<Service>,
    sum_capacity: f64,
}

impl<'a> PercLenModel<'a> {
    pub fn new(
        dc: &'a Datacentre,
        services: &'a Vec<Service>,
        sum_capacity: f64,
    ) -> PercLenModel<'a> {
        PercLenModel {
            dc,
            services,
            sum_capacity,
        }
    }

    pub fn evaluate(
        &self,
        // placements: &Solution<Vec<&Service>>,
        routes: &Vec<(ServiceID, Vec<RouteNode>)>,
    ) -> (f64, f64) {
        // Sum capacity of distinct VNFs
        let mut counted_vnfs = HashSet::new();
        let mut capacity_used = 0;

        for (service_id, route) in routes {
            for node in route {
                if let NodeType::VNF(server_id, stage) = node.node_type {
                    if counted_vnfs.contains(&(server_id, service_id, stage)) {
                        continue;
                    }

                    let vnf = &self.services[*service_id].vnfs[stage];
                    capacity_used = capacity_used + vnf.size;
                    counted_vnfs.insert((server_id, service_id, stage));
                }
            }
        }

        let perc_used = capacity_used as f64 / self.sum_capacity;

        let sum_len = routes.iter().map(|(_, route)| route.len()).sum::<usize>();
        let avg_len: f64 = sum_len as f64 / routes.len() as f64;

        (perc_used, avg_len)
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
        let service_a = Service {
            id: 0,
            prod_rate: 0.0,
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
            prod_rate: 0.0,
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

        let model = PercLenModel::new(&dc, &services, 1600.0);

        let mut simple_route = Vec::new();
        simple_route.push((
            0,
            vec![
                route_node(NodeType::VNF(0, 0)),
                route_node(NodeType::Component(0)),
                route_node(NodeType::Component(1)),
                route_node(NodeType::Component(2)),
                route_node(NodeType::Component(3)),
                route_node(NodeType::VNF(1, 1)),
            ],
        ));

        let mut revisited_route = Vec::new();
        revisited_route.push((
            0,
            vec![
                route_node(NodeType::VNF(0, 0)),
                route_node(NodeType::Component(0)),
                route_node(NodeType::Component(1)),
                route_node(NodeType::Component(2)),
                route_node(NodeType::Component(3)),
                route_node(NodeType::VNF(1, 1)),
                route_node(NodeType::Component(1)),
                route_node(NodeType::Component(3)),
                route_node(NodeType::Component(5)),
                route_node(NodeType::VNF(0, 1)),
            ],
        ));

        let mut multiple_services = Vec::new();
        multiple_services.push((
            0,
            vec![
                route_node(NodeType::VNF(0, 0)),
                route_node(NodeType::Component(0)),
                route_node(NodeType::Component(1)),
                route_node(NodeType::Component(2)),
                route_node(NodeType::Component(3)),
                route_node(NodeType::VNF(1, 1)),
            ],
        ));
        multiple_services.push((
            1,
            vec![
                route_node(NodeType::VNF(0, 0)),
                route_node(NodeType::Component(0)),
                route_node(NodeType::Component(1)),
                route_node(NodeType::Component(2)),
                route_node(NodeType::Component(3)),
                route_node(NodeType::Component(4)),
                route_node(NodeType::Component(5)),
                route_node(NodeType::VNF(1, 1)),
            ],
        ));

        let mut multiple_routes = Vec::new();
        multiple_routes.push((
            0,
            vec![
                route_node(NodeType::VNF(0, 0)),
                route_node(NodeType::Component(0)),
                route_node(NodeType::Component(1)),
                route_node(NodeType::Component(2)),
                route_node(NodeType::Component(3)),
                route_node(NodeType::VNF(1, 1)),
            ],
        ));
        multiple_routes.push((
            0,
            vec![
                route_node(NodeType::VNF(0, 0)),
                route_node(NodeType::Component(0)),
                route_node(NodeType::Component(1)),
                route_node(NodeType::Component(2)),
                route_node(NodeType::Component(3)),
                route_node(NodeType::Component(4)),
                route_node(NodeType::Component(5)),
                route_node(NodeType::VNF(3, 1)),
            ],
        ));

        let (perc_used_simp, avg_len_simp) = model.evaluate(&simple_route);
        let (perc_used_rev, avg_len_rev) = model.evaluate(&revisited_route);
        let (perc_used_mult, avg_len_mult) = model.evaluate(&multiple_services);
        let (perc_used_multr, avg_len_multr) = model.evaluate(&multiple_routes);

        // k = 4 : 16 servers
        assert_eq!(perc_used_simp, 100.0 / 1600.0);
        assert_eq!(avg_len_simp, 6.0);

        assert_eq!(perc_used_rev, 150.0 / 1600.0);
        assert_eq!(avg_len_rev, 10.0);

        assert_eq!(perc_used_mult, 170.0 / 1600.0);
        assert_eq!(avg_len_mult, 7.0);

        assert_eq!(perc_used_multr, 150.0 / 1600.0);
        assert_eq!(avg_len_multr, 7.0);
    }

    fn route_node(rn: NodeType) -> RouteNode {
        RouteNode {
            node_type: rn,
            route_count: 1,
            next_nodes: Vec::new(),
        }
    }
}
