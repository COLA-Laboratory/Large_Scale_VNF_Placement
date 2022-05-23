use std::{ops::Range, path::PathBuf};

use clap::{load_yaml, App};

use crate::{models::datacentre::Topology, operators::placement_strategies::FirstFit};

#[derive(Clone)]
pub struct Args {
    pub test_id: Option<usize>,
    pub run_range: Range<usize>,
    pub output_folder: PathBuf,
    pub max_evaluations: usize,
    pub topologies: Vec<Topology>,
    pub test_sizes: Vec<usize>,
    pub num_runs: usize,
    pub node_selection: FirstFit,
    pub perc_nearest: f64,
    pub ppls: PplsArgs,
    pub model: ModelArgs,
    pub problem: ProblemArgs,
    pub split_part: usize,
    pub num_split: usize,
}

impl Args {
    pub fn run_range(&self) -> Range<usize> {
        self.run_range.clone()
    }

    pub fn get_perc_nearest(&self, topology: &Topology, num_servers: usize) -> f64 {
        match topology {
            Topology::FatTree => match num_servers {
                500 => 0.18,
                1000 => 0.11,
                2000 => 0.08,
                4000 => 0.06,
                8000 => 0.04,
                16000 => 0.03,
                32000 => 0.03,
                64000 => 0.02,
                _ => 0.01,
            },
            Topology::LeafSpine => match num_servers {
                500 => 0.13,
                1000 => 0.08,
                2000 => 0.05,
                4000 => 0.03,
                8000 => 0.02,
                _ => 0.01,
            },
            Topology::DCell => match num_servers {
                500 => 0.33,
                1000 => 0.25,
                2000 => 0.18,
                4000 => 0.12,
                8000 => 0.08,
                16000 => 0.04,
                32000 => 0.04,
                64000 => 0.03,
                _ => 0.01,
            },
        }
    }
}

#[derive(Clone, Copy)]
pub struct PplsArgs {
    pub per_ind_evaluations: usize,
    pub num_weights: usize,
}

#[derive(Clone, Copy)]
pub struct ModelArgs {
    pub accuracy: f64,
    pub converged_iterations: usize,
    pub port_sr: f64,
    pub port_ql: usize,
    pub active_cost: f64,
    pub idle_cost: f64,
}
#[derive(Clone, Copy)]
pub struct ProblemArgs {
    pub mean_service_len: usize,
    pub variance_service_len: usize,
    pub max_service_len: usize,
    pub min_service_len: usize,
    pub mean_prod_rate: f64,
    pub variance_prod_rate: f64,
    pub min_prod_rate: f64,
    pub mean_service_rate: f64,
    pub variance_service_rate: f64,
    pub min_service_rate: f64,
    pub vnf_queue_length: usize,
    pub mean_vnf_size: f64,
    pub variance_vnf_size: f64,
    pub utilisation: f64,
}

pub fn get_args() -> Args {
    // Config file where default information is stored
    let mut settings = config::Config::default();
    settings.merge(config::File::with_name("Config")).unwrap();

    // CLI for overrides
    let yaml = load_yaml!("cli.yml");
    let matches = App::from_yaml(yaml).get_matches();

    // ---- General ----
    let test_id: Option<usize> = match matches.value_of("test_id") {
        Some(test_id) => Some(test_id.parse().unwrap()),
        None => None,
    };

    let split_part: usize = match matches.value_of("split") {
        Some(evals) => evals.parse().unwrap(),
        None => 0,
    };

    let num_split: usize = settings.get("num_splits").unwrap();

    let num_runs: usize = match matches.value_of("num_runs") {
        Some(evals) => evals.parse().unwrap(),
        None => settings.get("num_runs").unwrap(),
    };

    let split_len = num_runs / num_split;
    let min_split = split_part * split_len;
    let max_split = min_split + split_len;

    let run_range = min_split..max_split;

    let output_folder: String = match matches.value_of("folder") {
        Some(folder) => folder.to_string(),
        None => settings.get("results_folder").unwrap(),
    };
    let output_folder: PathBuf = [output_folder].iter().collect();

    let max_evaluations: usize = match matches.value_of("evaluations") {
        Some(evals) => evals.parse().unwrap(),
        None => settings.get("num_evaluations").unwrap(),
    };

    let topologies: Vec<Topology> = match matches.value_of("topologies") {
        Some(topology) => match topology {
            "FatTree" => vec![Topology::FatTree],
            "DCell" => vec![Topology::DCell],
            "LeafSpine" => vec![Topology::LeafSpine],
            _ => panic!("Invalid topology provided"),
        },
        None => vec![Topology::FatTree, Topology::DCell, Topology::LeafSpine],
    };

    let node_selection = FirstFit::new();
    let perc_nearest: f64 = settings.get("perc_nearest").unwrap();

    let test_size_id: usize = match matches.value_of("test_sizes") {
        Some(evals) => evals.parse().unwrap(),
        None => settings.get("test_sizes").unwrap(),
    };

    let test_sizes: Vec<usize> = match test_size_id {
        1 => vec![500, 1000, 2000, 4000],
        2 => vec![8000, 16000],
        3 => vec![32000],
        4 => vec![64000],
        5 => vec![128000],
        6 => vec![256000],
        _ => vec![500, 1000, 2000, 4000, 8000, 16000, 32000, 64000],
    };

    // --- PPLS ---
    let per_ind_evaluations = settings.get("per_ind_evaluations").unwrap();
    let num_weights = settings.get("num_weights").unwrap();

    let ppls = PplsArgs {
        per_ind_evaluations,
        num_weights,
    };

    // --- Model ---
    let accuracy = settings.get("accuracy").unwrap();
    let converged_iterations = settings.get("converged_iterations").unwrap();
    let port_sr = settings.get("port_sr").unwrap();
    let port_ql = settings.get("port_ql").unwrap();
    let active_cost = settings.get("active_cost").unwrap();
    let idle_cost = settings.get("idle_cost").unwrap();

    let model = ModelArgs {
        accuracy,
        converged_iterations,
        port_sr,
        port_ql,
        active_cost,
        idle_cost,
    };

    // --- Problem ---
    let mean_service_len = settings.get("mean_service_len").unwrap();
    let variance_service_len = settings.get("variance_service_len").unwrap();
    let max_service_len = settings.get("max_service_len").unwrap();
    let min_service_len = settings.get("min_service_len").unwrap();

    let mean_prod_rate = settings.get("mean_prod_rate").unwrap();
    let variance_prod_rate = settings.get("variance_prod_rate").unwrap();
    let min_prod_rate = settings.get("min_prod_rate").unwrap();

    let mean_service_rate = settings.get("mean_service_rate").unwrap();
    let variance_service_rate = settings.get("variance_service_rate").unwrap();
    let min_service_rate = settings.get("min_service_rate").unwrap();

    let vnf_queue_length = settings.get("vnf_queue_length").unwrap();

    let mean_vnf_size = settings.get("mean_vnf_size").unwrap();
    let variance_vnf_size = settings.get("variance_vnf_size").unwrap();

    let utilisation = settings.get("utilisation").unwrap();

    let problem = ProblemArgs {
        mean_service_len,
        variance_service_len,
        max_service_len,
        min_service_len,
        mean_prod_rate,
        variance_prod_rate,
        min_prod_rate,
        mean_service_rate,
        variance_service_rate,
        min_service_rate,
        vnf_queue_length,
        mean_vnf_size,
        variance_vnf_size,
        utilisation,
    };

    Args {
        test_id,
        run_range,
        output_folder,
        max_evaluations,
        topologies,
        test_sizes,
        num_runs,
        node_selection,
        perc_nearest,
        ppls,
        model,
        problem,
        split_part,
        num_split,
    }
}
