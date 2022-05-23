use super::{
    distance_matrix::DistanceMatrix, mapping::RouteNode, placement_strategies::NodeSelection,
    solution::Constraint,
};
use crate::{
    models::{
        constant_model::ConstantModel, datacentre::Datacentre,
        iterative_queueing_model::IterativeQueueingModel, mm1_queueing_model::MM1QueueingModel,
        perc_len_model::PercLenModel, routing::RoutingTable, service::Service,
        utilisation_latency_model::UtilisationLatencyModel, utilisation_model::UtilisationModel,
    },
    utilities::metrics::mean,
};

// NOTE: Don't need to check used capacity of each server as the mapping process only places VNFs
//       if there is capacity

pub trait Evaluation {
    fn evaluate_ind(&self, routes: &Vec<(usize, Vec<RouteNode>)>) -> Constraint<Vec<f64>, usize>;
}

#[derive(Clone)]
pub struct QueueingEval<'a, N: NodeSelection> {
    capacities: &'a Vec<usize>,
    distance_matrix: &'a DistanceMatrix,
    pub queueing_model: IterativeQueueingModel<'a>,
    routing_tables: &'a Vec<RoutingTable>,
    services: &'a Vec<Service>,
    node_selection: N,
    pub use_hf_cnstr: bool,
}

impl<'a, N: NodeSelection> QueueingEval<'a, N> {
    pub fn new(
        queueing_model: IterativeQueueingModel<'a>,
        routing_tables: &'a Vec<RoutingTable>,
        distance_matrix: &'a DistanceMatrix,
        capacities: &'a Vec<usize>,
        services: &'a Vec<Service>,
        node_selection: N,
    ) -> QueueingEval<'a, N> {
        QueueingEval {
            capacities,
            distance_matrix,
            queueing_model,
            routing_tables,
            services,
            node_selection,
            use_hf_cnstr: true,
        }
    }
}

impl<NS: NodeSelection> Evaluation for QueueingEval<'_, NS> {
    fn evaluate_ind(&self, routes: &Vec<(usize, Vec<RouteNode>)>) -> Constraint<Vec<f64>, usize> {
        let num_unplaced = num_unplaced(&routes, &self.services);
        if num_unplaced > 0 {
            return if self.use_hf_cnstr {
                Constraint::Infeasible(num_unplaced)
            } else {
                Constraint::Infeasible(0)
            };
        }

        let (latencies, pls, energy) = self.queueing_model.evaluate(&self.services, &routes);

        let avg_latency = mean(&latencies);
        let avg_pl = mean(&pls);

        Constraint::Feasible(vec![avg_latency, avg_pl, energy])
    }
}

// --- MM1 Model
#[derive(Clone)]
pub struct MM1Eval<'a, N: NodeSelection> {
    capacities: Vec<usize>,
    distance_matrix: &'a DistanceMatrix,
    model: MM1QueueingModel<'a>,
    routing_tables: &'a Vec<RoutingTable>,
    services: &'a Vec<Service>,
    node_selection: &'a N,
}

impl<'a, N: NodeSelection> MM1Eval<'a, N> {
    pub fn new(
        dc: &'a Datacentre,
        routing_tables: &'a Vec<RoutingTable>,
        distance_matrix: &'a DistanceMatrix,
        capacities: Vec<usize>,
        services: &'a Vec<Service>,
        sw_sr: f64,
        active_cost: f64,
        idle_cost: f64,
        node_selection: &'a N,
    ) -> MM1Eval<'a, N> {
        let model = MM1QueueingModel::new(dc, sw_sr, active_cost, idle_cost);

        MM1Eval {
            capacities,
            distance_matrix,
            model,
            routing_tables,
            services,
            node_selection,
        }
    }
}

impl<NS: NodeSelection> Evaluation for MM1Eval<'_, NS> {
    fn evaluate_ind(&self, routes: &Vec<(usize, Vec<RouteNode>)>) -> Constraint<Vec<f64>, usize> {
        // -- Evaluate solution and check feasibility
        let num_unplaced = num_unplaced(&routes, &self.services);
        if num_unplaced > 0 {
            return Constraint::Infeasible(num_unplaced);
        }

        let (service_latency, energy) = self.model.evaluate(&self.services, &routes);

        let mut num_infeasible = 0;
        let mut mean_latency = 0.0;
        for latency in service_latency {
            if latency == std::f64::INFINITY {
                num_infeasible += 1;
            } else {
                mean_latency += latency;
            }
        }
        mean_latency = mean_latency / self.services.len() as f64;

        if num_infeasible > 0 {
            Constraint::Infeasible(num_infeasible)
        } else {
            Constraint::Feasible(vec![mean_latency, energy])
        }
    }
}

// --- Utilisation Model
#[derive(Clone)]
pub struct UtilisationEval<'a, N: NodeSelection> {
    capacities: Vec<usize>,
    distance_matrix: &'a DistanceMatrix,
    util_model: UtilisationModel<'a>,
    routing_tables: &'a Vec<RoutingTable>,
    services: &'a Vec<Service>,
    node_selection: &'a N,
}

impl<'a, N: NodeSelection> UtilisationEval<'a, N> {
    pub fn new(
        dc: &'a Datacentre,
        routing_tables: &'a Vec<RoutingTable>,
        distance_matrix: &'a DistanceMatrix,
        capacities: Vec<usize>,
        services: &'a Vec<Service>,
        sw_sr: f64,
        sw_ql: usize,
        node_selection: &'a N,
        queueing_model: IterativeQueueingModel<'a>,
    ) -> UtilisationEval<'a, N> {
        let util_model = UtilisationModel::new(dc, queueing_model, sw_sr, sw_ql);

        UtilisationEval {
            capacities,
            distance_matrix,
            util_model,
            routing_tables,
            services,
            node_selection,
        }
    }
}

impl<NS: NodeSelection> Evaluation for UtilisationEval<'_, NS> {
    fn evaluate_ind(&self, routes: &Vec<(usize, Vec<RouteNode>)>) -> Constraint<Vec<f64>, usize> {
        // -- Evaluate solution and check feasibility
        let num_unplaced = num_unplaced(&routes, &self.services);
        if num_unplaced > 0 {
            return Constraint::Infeasible(num_unplaced);
        }

        let (service_utilisation, energy) =
            self.util_model
                .evaluate(&self.services, &routes, |util| util);

        let avg_util = mean(&service_utilisation);

        Constraint::Feasible(vec![avg_util, energy])
    }
}

// --- Heuristic Models
// Percentage / Length model
#[derive(Clone)]
pub struct PercLenEval<'a, N: NodeSelection> {
    routing_tables: &'a Vec<RoutingTable>,
    distance_matrix: &'a DistanceMatrix,
    capacities: Vec<usize>,
    heuristic_model: PercLenModel<'a>,
    services: &'a Vec<Service>,
    node_selection: &'a N,
}

impl<'a, N: NodeSelection> PercLenEval<'a, N> {
    pub fn new(
        dc: &'a Datacentre,
        routing_tables: &'a Vec<RoutingTable>,
        distance_matrix: &'a DistanceMatrix,
        capacities: Vec<usize>,
        services: &'a Vec<Service>,
        node_selection: &'a N,
    ) -> PercLenEval<'a, N> {
        let sum_capacity = capacities.iter().sum::<usize>() as f64;
        let heuristic_model = PercLenModel::new(dc, services, sum_capacity);

        PercLenEval {
            routing_tables,
            distance_matrix,
            capacities,
            heuristic_model,
            services,
            node_selection,
        }
    }
}

impl<NS: NodeSelection> Evaluation for PercLenEval<'_, NS> {
    fn evaluate_ind(&self, routes: &Vec<(usize, Vec<RouteNode>)>) -> Constraint<Vec<f64>, usize> {
        // -- Evaluate solution and check feasibility
        let num_unplaced = num_unplaced(&routes, &self.services);
        if num_unplaced > 0 {
            return Constraint::Infeasible(num_unplaced);
        }

        let (perc_used, len) = self.heuristic_model.evaluate(&routes);

        Constraint::Feasible(vec![perc_used, len])
    }
}

// Utilisation Latency Model
#[derive(Clone)]
pub struct UtilisationLatencyEval<'a, N: NodeSelection> {
    routing_tables: &'a Vec<RoutingTable>,
    distance_matrix: &'a DistanceMatrix,
    capacities: Vec<usize>,
    heuristic_model: UtilisationLatencyModel<'a>,
    services: &'a Vec<Service>,
    node_selection: &'a N,
}

impl<'a, N: NodeSelection> UtilisationLatencyEval<'a, N> {
    pub fn new(
        dc: &'a Datacentre,
        routing_tables: &'a Vec<RoutingTable>,
        distance_matrix: &'a DistanceMatrix,
        capacities: Vec<usize>,
        services: &'a Vec<Service>,
        node_selection: &'a N,
        queueing_model: IterativeQueueingModel<'a>,
    ) -> UtilisationLatencyEval<'a, N> {
        let heuristic_model =
            UtilisationLatencyModel::new(dc, services, queueing_model, capacities.clone());

        UtilisationLatencyEval {
            routing_tables,
            distance_matrix,
            capacities,
            heuristic_model,
            services,
            node_selection,
        }
    }
}

impl<NS: NodeSelection> Evaluation for UtilisationLatencyEval<'_, NS> {
    fn evaluate_ind(&self, routes: &Vec<(usize, Vec<RouteNode>)>) -> Constraint<Vec<f64>, usize> {
        // -- Evaluate solution and check feasibility
        let num_unplaced = num_unplaced(&routes, &self.services);
        if num_unplaced > 0 {
            return Constraint::Infeasible(num_unplaced);
        }

        let (latencies, energy) = self.heuristic_model.evaluate(&routes);
        let mean_latency = latencies.iter().sum::<f64>() / latencies.len() as f64;

        Constraint::Feasible(vec![mean_latency, energy])
    }
}

// Utilisation Latency Model
#[derive(Clone)]
pub struct ConstantEval<'a, N: NodeSelection> {
    routing_tables: &'a Vec<RoutingTable>,
    distance_matrix: &'a DistanceMatrix,
    capacities: Vec<usize>,
    heuristic_model: ConstantModel<'a>,
    services: &'a Vec<Service>,
    node_selection: &'a N,
}

impl<'a, N: NodeSelection> ConstantEval<'a, N> {
    pub fn new(
        dc: &'a Datacentre,
        routing_tables: &'a Vec<RoutingTable>,
        distance_matrix: &'a DistanceMatrix,
        capacities: Vec<usize>,
        services: &'a Vec<Service>,
        node_selection: &'a N,
        queueing_model: IterativeQueueingModel<'a>,
    ) -> ConstantEval<'a, N> {
        let heuristic_model = ConstantModel::new(dc, services, queueing_model, 1.0, 0.01);

        ConstantEval {
            routing_tables,
            distance_matrix,
            capacities,
            heuristic_model,
            services,
            node_selection,
        }
    }
}

impl<NS: NodeSelection> Evaluation for ConstantEval<'_, NS> {
    fn evaluate_ind(&self, routes: &Vec<(usize, Vec<RouteNode>)>) -> Constraint<Vec<f64>, usize> {
        // -- Evaluate solution and check feasibility
        let num_unplaced = num_unplaced(&routes, &self.services);
        if num_unplaced > 0 {
            return Constraint::Infeasible(num_unplaced);
        }

        let (latencies, pls, energy) = self.heuristic_model.evaluate(&routes);
        let mean_latency = latencies.iter().sum::<f64>() / latencies.len() as f64;
        let mean_pl = pls.iter().sum::<f64>() / pls.len() as f64;

        Constraint::Feasible(vec![mean_latency, mean_pl, energy])
    }
}

// --- Helpers
pub fn num_unplaced(routes: &Vec<(usize, Vec<RouteNode>)>, services: &Vec<Service>) -> usize {
    let mut counts = vec![0; services.len()];
    routes.iter().for_each(|&(s_id, _)| {
        counts[s_id] += 1;
    });

    counts.iter().filter(|&&count| count == 0).count()
}
