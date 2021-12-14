mod algorithms;
mod models;
mod operators;
mod utilities;

use algorithms::{moead, ppls_simple, pplsd};
use distance_matrix::{build_cache, DistanceMatrix};
use models::{
    datacentre::{DCell, Datacentre, FatTree, LeafSpine, Topology},
    iterative_queueing_model::IterativeQueueingModel,
    routing::{self},
    service::{Service, VNF},
};
use operators::{
    crossover::UniformCrossover,
    distance_matrix::{self},
    evaluation::{
        num_unplaced, ConstantEval, Evaluation, MM1Eval, PercLenEval, QueueingEval,
        UtilisationEval, UtilisationLatencyEval,
    },
    initialisation::{ImprovedServiceAwareInitialisation, InitPop, ServiceAwareInitialisation},
    mapping::{Mapping, ServiceToRouteMapping},
    neighbour_gen::AddRemSwap,
    placement_strategies::FirstFit,
    solution::{Constraint, Solution},
};
use rand::prelude::*;
use rand_distr::{Distribution, Normal};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use routing::RoutingTable;
use std::{
    fs::{self, File, OpenOptions},
    io::{prelude::*, BufReader, BufWriter, LineWriter},
    path::{Path, PathBuf},
};

use utilities::{args::get_args, args::Args, stopwatch::Stopwatch};

use crate::{
    algorithms::{ibea, nsgaii},
    operators::distance_matrix::build_cache_dtm,
};

fn main() {
    let args: Args = get_args();

    match args.test_id {
        Some(0) => create_topologies(&args),
        Some(1) => create_topologies_uncompressed(&args),
        Some(2) => run_solution_construction_tests(&args),
        Some(3) => run_initialization_tests(&args),
        Some(4) => run_algorithm_comparison_tests(&args),
        Some(5) => run_model_tests(&args),
        Some(6) => run_vlg_tests(&args),
        _ => {
            // count_topology_rows();

            // create_topologies(&args);
            // create_topologies_uncompressed(&args);

            // run_solution_construction_tests(&args);

            // run_initialization_tests(&args);
            // run_algorithm_comparison_tests(&args);
            // run_model_tests(&args);

            run_vlg_tests(&args);
        }
    }
}

fn count_topology_rows() {
    let sizes = [
        500, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000, 256000,
    ];

    let out_folder = "topology_rows";
    let file = File::create(out_folder).unwrap();
    let mut line_writer = LineWriter::new(file);

    for topology in &[Topology::DCell, Topology::FatTree, Topology::LeafSpine] {
        println!("{:?}", topology);

        line_writer
            .write_all(format!("{:?} \n", topology).as_bytes())
            .unwrap();

        for size in &sizes {
            let topology = load_topology(&topology, *size);

            if topology.is_none() {
                continue;
            }

            let (_, rows) = topology.unwrap();
            let total_rows: usize = rows.iter().map(|rt| rt.len()).sum();

            line_writer
                .write_all(format!("{} {} \n", size, total_rows).as_bytes())
                .unwrap();
        }
    }
}

/**
    Creates the commonly used topologies and saves them to use later.
    Saves time rather than recreating the same datacentres multiple times.

    Datacenter parameters have been set to have approximately the same number of servers for each scale.
**/
fn create_topologies(args: &Args) {
    println!("Creating topologies");

    let sizes = [
        500, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000, 256000,
    ];

    let base_folder = Path::new("topology").join("compressed");
    fs::create_dir_all(&base_folder).unwrap();

    // -- Fat Tree --
    if args.topologies.contains(&Topology::FatTree) {
        for (i, k) in [12, 16, 20, 24, 32, 40, 52, 64, 80, 100].iter().enumerate() {
            println!("Fat Tree: ({},{})", i, k);

            let dc = FatTree::new(*k);

            let (bgn, end, routing_tables) =
                load_table_part(args, &Topology::FatTree, sizes[i], &dc);

            let routing_tables = routing::get_tables(&dc, true, bgn, end, routing_tables);

            println!("Completed: FT {}", dc.num_servers);
            write_topology(
                Topology::FatTree,
                &base_folder,
                sizes[i],
                &dc,
                &routing_tables,
            );
        }
    }

    // -- Leaf Spine --
    if args.topologies.contains(&Topology::LeafSpine) {
        for (i, (num_ports, num_spine)) in [
            (32, 16),
            (44, 22),
            (64, 32),
            (90, 45),
            (126, 63),
            (178, 89),
            (252, 126),
            (358, 179),
            (506, 253),
            (714, 357),
        ]
        .iter()
        .enumerate()
        {
            println!("Leaf Spine: ({},{})", num_ports, num_spine);

            let dc = LeafSpine::new(*num_ports, *num_spine);

            let (bgn, end, routing_tables) =
                load_table_part(args, &Topology::LeafSpine, sizes[i], &dc);

            let routing_tables = routing::get_tables(&dc, true, bgn, end, routing_tables);

            println!("Completed: LS {}", dc.num_servers);
            write_topology(
                Topology::LeafSpine,
                &base_folder,
                sizes[i],
                &dc,
                &routing_tables,
            );
        }
    }

    // -- DCell --
    if args.topologies.contains(&Topology::DCell) {
        for (i, num_ports) in [4, 5, 6, 7, 9, 11, 13, 15, 18, 22].iter().enumerate() {
            println!("DCell: {}", num_ports);

            let dc = DCell::new(*num_ports, 2);

            // For very large DCs we have to create the topology in parts
            let (bgn, end, routing_tables) = load_table_part(args, &Topology::DCell, sizes[i], &dc);

            let routing_tables = routing::get_tables(&dc, true, bgn, end, routing_tables);

            println!("Completed: DC {}", dc.num_servers);
            write_topology(
                Topology::DCell,
                &base_folder,
                sizes[i],
                &dc,
                &routing_tables,
            );
        }
    }
}

fn load_table_part(
    args: &Args,
    topology: &Topology,
    size: usize,
    dc: &Datacentre,
) -> (usize, usize, Vec<RoutingTable>) {
    let num_servers = dc.num_servers;
    let num_components = dc.num_components();

    // For very large table sizes we have to create the tables in parts
    if size < 128000 {
        return (
            0,
            dc.num_servers,
            vec![RoutingTable::new(); dc.num_components()],
        );
    }

    let split_part = args.split_part;

    let split_length = ((num_servers as f64) / (args.num_split as f64)).ceil() as usize;
    let bgn = split_length * split_part;
    let end = (bgn + split_length).min(num_servers);

    let topo = load_topology(topology, size);

    let rt = if let Some((_, rt)) = topo {
        rt
    } else {
        vec![RoutingTable::new(); num_components]
    };

    (bgn, end, rt)
}

fn create_topologies_uncompressed(args: &Args) {
    println!("Creating topologies");

    let sizes = [500, 1000, 2000, 4000, 8000, 16000, 32000, 64000];

    let base_folder = args.output_folder.clone().join("Uncompressed");
    fs::create_dir_all(&base_folder).unwrap();

    // -- Fat Tree --
    if args.topologies.contains(&Topology::FatTree) {
        for (i, k) in [12, 16, 20, 24, 32, 40, 52, 64].iter().enumerate() {
            println!("Fat Tree: ({},{})", i, k);

            let dc = FatTree::new(*k);
            let (bgn, end, routing_tables) =
                load_table_part(args, &Topology::FatTree, sizes[i], &dc);

            let routing_tables = routing::get_tables(&dc, false, bgn, end, routing_tables);

            println!("Completed: FT {}", dc.num_servers);
            write_topology(
                Topology::FatTree,
                &base_folder,
                sizes[i],
                &dc,
                &routing_tables,
            );
        }
    }

    // -- Leaf Spine --
    if args.topologies.contains(&Topology::LeafSpine) {
        for (i, (num_ports, num_spine)) in [
            (32, 16),
            (44, 22),
            (64, 32),
            (90, 45),
            (126, 63),
            (178, 89),
            (252, 126),
            (358, 179),
        ]
        .iter()
        .enumerate()
        {
            println!("Leaf Spine: ({},{})", num_ports, num_spine);

            let dc = LeafSpine::new(*num_ports, *num_spine);
            let (bgn, end, routing_tables) =
                load_table_part(args, &Topology::LeafSpine, sizes[i], &dc);

            let routing_tables = routing::get_tables(&dc, false, bgn, end, routing_tables);

            println!("Completed: LS {}", dc.num_servers);
            write_topology(
                Topology::LeafSpine,
                &base_folder,
                sizes[i],
                &dc,
                &routing_tables,
            );
        }
    }

    // -- DCell --
    if args.topologies.contains(&Topology::DCell) {
        for (i, num_ports) in [4, 5, 6, 7, 9, 11, 13, 15].iter().enumerate() {
            println!("DCell: {}", num_ports);

            let dc = DCell::new(*num_ports, 2);
            let (bgn, end, routing_tables) = load_table_part(args, &Topology::DCell, sizes[i], &dc);

            let routing_tables = routing::get_tables(&dc, false, bgn, end, routing_tables);

            println!("Completed: DC {}", dc.num_servers);
            write_topology(
                Topology::DCell,
                &base_folder,
                sizes[i],
                &dc,
                &routing_tables,
            );
        }
    }
}

fn run_solution_construction_tests(args: &Args) {
    println!("Solution Construction");

    let output_folder = args.output_folder.clone();
    let test_folder = output_folder.join("SolutionConstruction");

    // Custom args to change VNF size
    let mut args = args.clone();

    for topology in args.topologies.iter() {
        for scale in &args.test_sizes {
            let (dc, rt) = load_topology(topology, *scale).unwrap();

            args.problem.mean_vnf_size = 100.0;
            args.problem.variance_vnf_size = 0.0;

            (0..100).into_iter().for_each(|perc: i32| {
                let perc_nearest = perc as f64 / 100.0;
                let num_nearest = ((dc.num_servers as f64) * perc_nearest).ceil() as usize;

                let capacities = vec![100; dc.num_servers];

                // Caching techniques
                let rng_dm = build_cache(&dc, num_nearest);
                let dtm_dm = build_cache_dtm(&dc, num_nearest);

                // Mapping technique
                let rng_mapping =
                    ServiceToRouteMapping::new(FirstFit::new(), &capacities, &rng_dm, &rt);

                let dtm_mapping =
                    ServiceToRouteMapping::new(FirstFit::new(), &capacities, &dtm_dm, &rt);

                let utilisation = 0.95;

                let mut rng_num_feasible = 0;
                let mut dtm_num_feasible = 0;

                for t in 0..1000 {
                    // Prepare problem instance
                    let services = create_problem_instance(&args, utilisation, dc.num_servers);
                    let init = ServiceAwareInitialisation::new(&services, dc.num_servers);
                    let test_pop = init.apply(100);

                    for ind in &test_pop {
                        let rng_routes = rng_mapping.apply(&ind);
                        if num_unplaced(&rng_routes, &services) == 0 {
                            rng_num_feasible += 1;
                        }

                        let dtm_routes = dtm_mapping.apply(&ind);
                        if num_unplaced(&dtm_routes, &services) == 0 {
                            dtm_num_feasible += 1;
                        }
                    }
                }

                // Print means
                let out_folder = test_folder
                    .join(topology.to_string())
                    .join(scale.to_string());

                append_to_file(
                    &out_folder,
                    format!("{}.csv", perc),
                    &format!("{},{}\n", rng_num_feasible, dtm_num_feasible),
                );
            });
        }
    }
}

fn run_initialization_tests(args: &Args) {
    println!("Starting initialization tests");

    let output_folder = args.output_folder.clone();
    let test_folder = output_folder.join("Initialization");

    let num_weights = 128;

    for topology in args.topologies.iter() {
        for scale in &args.test_sizes {
            let (dc, rt) = load_topology(topology, *scale).unwrap();

            // Preparing test
            let num_servers = dc.num_servers;

            let num_nearest = (num_servers as f64) * args.get_perc_nearest(topology, *scale);
            let dm = build_cache(&dc, num_nearest.ceil() as usize);

            let capacities = vec![100; dc.num_servers];

            (0..30).into_par_iter().for_each(|run: i32| {
                let services = create_problem_instance(args, args.problem.utilisation, num_servers);

                let sa_init = ServiceAwareInitialisation::new(&services, dc.num_servers);
                let isa_init = ImprovedServiceAwareInitialisation::new(&services, dc.num_servers);

                let mapping = get_default_mapping(&args, &capacities, &dm, &rt);
                let model = get_default_model(args, &services, &dc, &capacities, &rt, &dm);

                // Running test
                let sa_pop = sa_init.apply(num_weights);
                let isa_pop = isa_init.apply(num_weights);

                // Reporting results
                let sa_objectives = sa_pop
                    .iter()
                    .map(|ind| {
                        let routes = mapping.apply(ind);
                        model.evaluate_ind(&routes)
                    })
                    .collect();

                let isa_objectives = isa_pop
                    .iter()
                    .map(|ind| {
                        let routes = mapping.apply(ind);
                        model.evaluate_ind(&routes)
                    })
                    .collect();

                let out_folder = test_folder
                    .join(topology.to_string())
                    .join(scale.to_string())
                    .join(run.to_string());

                print_objectives(
                    &out_folder.join("sa"),
                    format!("{}_{}.objs", services.len(), num_weights),
                    &sa_objectives,
                )
                .unwrap();

                print_objectives(
                    &out_folder.join("isa"),
                    format!("{}_{}.objs", services.len(), num_weights),
                    &isa_objectives,
                )
                .unwrap();
            });
        }
    }
}

fn run_algorithm_comparison_tests(args: &Args) {
    println!("Starting algorithm comparison tests");

    let output_folder = args.output_folder.clone();
    let test_folder = output_folder.join("AlgorithmComparison");

    let max_evaluations = args.max_evaluations;

    for topology in args.topologies.iter() {
        for scale in &args.test_sizes {
            let (dc, rt) = load_topology(topology, *scale).unwrap();

            // Preparing test
            let num_servers = dc.num_servers;
            let num_nearest = (num_servers as f64) * args.perc_nearest;
            let dm = build_cache(&dc, num_nearest.ceil() as usize);

            let capacities = vec![100; dc.num_servers];

            for problem_instance in args.run_range() {
                let test_folder = test_folder
                    .join(topology.to_string())
                    .join(scale.to_string())
                    .join(problem_instance.to_string());

                let services = create_problem_instance(args, args.problem.utilisation, num_servers);
                let vnfs = services.iter().map(|s| s).collect();

                let mapping = get_default_mapping(&args, &capacities, &dm, &rt);
                let neighbour_gen = AddRemSwap::new(vnfs);

                let init_pop = get_default_initialisation(&services, &dc);

                let model = get_default_model(&args, &services, &dc, &capacities, &rt, &dm);

                // Proposed algorithm
                let alg_folder = test_folder.join("PPLS_Simple");

                let mut ppls_simple_timer = Stopwatch::new();
                ppls_simple_timer.start();

                println!("PPLS SIMPLE");
                ppls_simple::run(
                    &init_pop,
                    &mapping,
                    &model,
                    neighbour_gen.clone(),
                    128,
                    max_evaluations,
                    3,
                    false,
                    |evaluations, pop| {
                        ppls_simple_timer.pause();

                        print_population_objectives(
                            &alg_folder,
                            format!("{}_{}.objs", services.len(), evaluations),
                            &pop,
                        )
                        .unwrap();

                        ppls_simple_timer.start();
                    },
                );

                ppls_simple_timer.pause();

                let mut file = get_file(&alg_folder, "running_time.out");

                write!(file, "{}", ppls_simple_timer.read()).unwrap();

                // PPLS/D
                println!("PPLS/D");
                let alg_folder = test_folder.join("PPLSD");

                let mut ppls_timer = Stopwatch::new();
                ppls_timer.start();

                pplsd::run(
                    &init_pop,
                    &mapping,
                    &model,
                    neighbour_gen.clone(),
                    16,
                    max_evaluations,
                    10,
                    3,
                    |evaluations, pop| {
                        ppls_timer.pause();

                        print_population_objectives(
                            &alg_folder,
                            format!("{}_{}.objs", services.len(), evaluations),
                            &pop,
                        )
                        .unwrap();

                        ppls_timer.start();
                    },
                );

                ppls_timer.pause();

                let mut file = get_file(&alg_folder, "running_time.out");

                write!(file, "{}", ppls_timer.read()).unwrap();

                // MOEA/D
                println!("MOEA/D");
                let alg_folder = test_folder.join("MOEAD");
                let crossover = UniformCrossover::new(0.4);

                let mut moead_timer = Stopwatch::new();
                moead_timer.start();

                moead::run(
                    &init_pop,
                    &mapping,
                    &model,
                    neighbour_gen.clone(),
                    &crossover,
                    3,
                    max_evaluations,
                    128,
                    13,
                    |evaluations, pop| {
                        moead_timer.pause();

                        print_population_objectives(
                            &alg_folder,
                            format!("{}_{}.objs", services.len(), evaluations),
                            &pop,
                        )
                        .unwrap();

                        moead_timer.start();
                    },
                );

                moead_timer.pause();

                let mut file = get_file(&alg_folder, "running_time.out");

                write!(file, "{}", moead_timer.read()).unwrap();

                // IBEA
                println!("IBEA");
                let alg_folder = test_folder.join("IBEA");
                let crossover = UniformCrossover::new(0.4);

                let mut ibea_timer = Stopwatch::new();
                ibea_timer.start();

                ibea::run(
                    &init_pop,
                    &mapping,
                    &model,
                    neighbour_gen.clone(),
                    &crossover,
                    max_evaluations,
                    128,
                    0.05,
                    |evaluations, pop| {
                        ibea_timer.pause();

                        print_population_objectives(
                            &alg_folder,
                            format!("{}_{}.objs", services.len(), evaluations),
                            &pop,
                        )
                        .unwrap();

                        ibea_timer.start();
                    },
                );

                ibea_timer.pause();

                let mut file = get_file(&alg_folder, "running_time.out");

                write!(file, "{}", ibea_timer.read()).unwrap();

                // NSGA-II
                println!("NSGA-II");
                let alg_folder = test_folder.join("NSGA-II");
                let crossover = UniformCrossover::new(0.4);

                let mut nsgaii_timer = Stopwatch::new();
                nsgaii_timer.start();

                nsgaii::run(
                    &init_pop,
                    &mapping,
                    &model,
                    neighbour_gen.clone(),
                    &crossover,
                    max_evaluations,
                    128,
                    |evaluations, pop| {
                        nsgaii_timer.pause();

                        // for ind in  pop {
                        //     println!("{:?}", ind.objectives);
                        // }

                        print_population_objectives(
                            &alg_folder,
                            format!("{}_{}.objs", services.len(), evaluations),
                            &pop,
                        )
                        .unwrap();

                        nsgaii_timer.start();
                    },
                );

                nsgaii_timer.pause();

                let mut file = get_file(&alg_folder, "running_time.out");

                write!(file, "{}", nsgaii_timer.read()).unwrap();
            }
        }
    }
}

fn run_model_tests(args: &Args) {
    println!("Starting model tests");

    let output_folder = args.output_folder.clone();
    let test_folder = output_folder.join("Model");

    let accuracy = args.model.accuracy;
    let converged_iterations = args.model.converged_iterations;
    let active_cost = args.model.active_cost;
    let idle_cost = args.model.idle_cost;

    let max_evaluations = args.max_evaluations;

    for topology in args.topologies.iter() {
        for scale in &args.test_sizes {
            let (dc, rt) = load_topology(topology, *scale).unwrap();

            let num_servers = dc.num_servers;
            let num_nearest = (num_servers as f64) * args.perc_nearest;
            let dm = build_cache(&dc, num_nearest.ceil() as usize);

            // Common parameters
            let capacities = vec![100; dc.num_servers];

            let sw_sr = args.model.port_sr * dc.num_ports as f64;
            let sw_ql = args.model.port_ql * dc.num_ports;

            let node_selection = args.node_selection.clone();

            for problem_instance in args.run_range() {
                let services = create_problem_instance(args, args.problem.utilisation, num_servers);
                let vnfs = services.iter().map(|s| s).collect();

                let mapping = get_default_mapping(&args, &capacities, &dm, &rt);
                let neighbour_gen = AddRemSwap::new(vnfs);

                let init_pop = get_default_initialisation(&services, &dc);

                // For comparison
                let acc_qm = IterativeQueueingModel::new(
                    &dc,
                    sw_sr,
                    sw_ql,
                    accuracy,
                    converged_iterations,
                    active_cost,
                    idle_cost,
                );
                let acc_qe = QueueingEval::new(
                    acc_qm.clone(),
                    &rt,
                    &dm,
                    &capacities,
                    &services,
                    node_selection.clone(),
                );

                let out_folder = test_folder
                    .join(topology.to_string())
                    .join(scale.to_string())
                    .join(problem_instance.to_string());
                /* Different levels of iterative model accuracy */
                for (model_accuracy, converged_iterations) in [
                    (std::f64::INFINITY, 1), // Case with no iterations
                    (500.0, converged_iterations),
                    (50.0, converged_iterations),
                    (5.0, converged_iterations),
                    (0.5, converged_iterations),
                ]
                .iter()
                {
                    println!("{} {} {}", topology, scale, model_accuracy);

                    let cmp_qm = IterativeQueueingModel::new(
                        &dc,
                        sw_sr,
                        sw_ql,
                        *model_accuracy,
                        *converged_iterations,
                        active_cost,
                        idle_cost,
                    );
                    let cmp_qe = QueueingEval::new(
                        cmp_qm,
                        &rt,
                        &dm,
                        &capacities,
                        &services,
                        node_selection.clone(),
                    );

                    let mut model_timer = Stopwatch::new();
                    model_timer.start();

                    ppls_simple::run(
                        &init_pop,
                        &mapping,
                        &cmp_qe,
                        neighbour_gen.clone(),
                        128,
                        max_evaluations,
                        3,
                        false,
                        |evaluations, pop| {
                            model_timer.pause();

                            let mut new_pop = pop.clone();

                            for ind in &mut new_pop {
                                let routes = mapping.apply(ind);
                                ind.objectives = acc_qe.evaluate_ind(&routes);
                            }

                            print_population_objectives(
                                &out_folder.join(model_accuracy.to_string()),
                                format!("{}_{}.objs", services.len(), evaluations),
                                &new_pop,
                            )
                            .unwrap();

                            model_timer.start();
                        },
                    );

                    model_timer.pause();

                    let mut file = get_file(
                        &out_folder.join(model_accuracy.to_string()),
                        "running_time.out",
                    );

                    write!(file, "{}", model_timer.read()).unwrap();
                }

                /* M/M/1 Model */
                println!("{} {} MM1", topology, scale);

                let mm1_eval = MM1Eval::new(
                    &dc,
                    &rt,
                    &dm,
                    capacities.clone(),
                    &services,
                    sw_sr,
                    active_cost,
                    idle_cost,
                    &node_selection,
                );

                let mut model_timer = Stopwatch::new();
                model_timer.start();

                ppls_simple::run(
                    &init_pop,
                    &mapping,
                    &mm1_eval,
                    neighbour_gen.clone(),
                    128,
                    max_evaluations,
                    2,
                    false,
                    |evaluations, pop| {
                        model_timer.pause();

                        let mut new_pop = pop.clone();

                        for ind in &mut new_pop {
                            let routes = mapping.apply(ind);
                            ind.objectives = acc_qe.evaluate_ind(&routes);
                        }

                        print_population_objectives(
                            &out_folder.join("MM1Model"),
                            format!("{}_{}.objs", services.len(), evaluations),
                            &new_pop,
                        )
                        .unwrap();

                        model_timer.start();
                    },
                );

                model_timer.pause();

                let mut file = get_file(&out_folder.join("MM1Model"), "running_time.out");

                write!(file, "{}", model_timer.read()).unwrap();

                /* Utilisation Model */
                println!("{} {} Utilisation", topology, scale);

                let qm_base = IterativeQueueingModel::new(
                    &dc,
                    sw_sr,
                    sw_ql,
                    std::f64::INFINITY,
                    1,
                    active_cost,
                    idle_cost,
                );

                let ue_eval = UtilisationEval::new(
                    &dc,
                    &rt,
                    &dm,
                    capacities.clone(),
                    &services,
                    sw_sr,
                    sw_ql,
                    &node_selection,
                    qm_base,
                );

                let mut model_timer = Stopwatch::new();
                model_timer.start();

                ppls_simple::run(
                    &init_pop,
                    &mapping,
                    &ue_eval,
                    neighbour_gen.clone(),
                    128,
                    max_evaluations,
                    2,
                    false,
                    |evaluations, pop| {
                        model_timer.pause();

                        let mut new_pop = pop.clone();

                        for ind in &mut new_pop {
                            let routes = mapping.apply(ind);
                            ind.objectives = acc_qe.evaluate_ind(&routes);
                        }

                        print_population_objectives(
                            &out_folder.join("UtilisationModel"),
                            format!("{}_{}.objs", services.len(), evaluations),
                            &new_pop,
                        )
                        .unwrap();

                        model_timer.start();
                    },
                );

                model_timer.pause();

                let mut file = get_file(&out_folder.join("UtilisationModel"), "running_time.out");

                write!(file, "{}", model_timer.read()).unwrap();

                // Percentage / Length Model
                println!("{} {} Perc. Len", topology, scale);

                let perc_len_eval = PercLenEval::new(
                    &dc,
                    &rt,
                    &dm,
                    capacities.clone(),
                    &services,
                    &node_selection,
                );

                let mut model_timer = Stopwatch::new();
                model_timer.start();

                ppls_simple::run(
                    &init_pop,
                    &mapping,
                    &perc_len_eval,
                    neighbour_gen.clone(),
                    128,
                    max_evaluations,
                    2,
                    false,
                    |evaluations, pop| {
                        model_timer.pause();

                        let mut new_pop = pop.clone();

                        for ind in &mut new_pop {
                            let routes = mapping.apply(ind);
                            ind.objectives = acc_qe.evaluate_ind(&routes);
                        }

                        print_population_objectives(
                            &out_folder.join("PercLenModel"),
                            format!("{}_{}.objs", services.len(), evaluations),
                            &new_pop,
                        )
                        .unwrap();

                        model_timer.start();
                    },
                );

                model_timer.pause();

                let mut file = get_file(&out_folder.join("PercLenModel"), "running_time.out");

                write!(file, "{}", model_timer.read()).unwrap();

                // Constant waiting time
                println!("{} {} Constant", topology, scale);

                let constant_eval = ConstantEval::new(
                    &dc,
                    &rt,
                    &dm,
                    capacities.clone(),
                    &services,
                    &node_selection,
                    acc_qm.clone(),
                );

                let mut model_timer = Stopwatch::new();
                model_timer.start();

                ppls_simple::run(
                    &init_pop,
                    &mapping,
                    &constant_eval,
                    neighbour_gen.clone(),
                    128,
                    max_evaluations,
                    3,
                    false,
                    |evaluations, pop| {
                        model_timer.pause();

                        let mut new_pop = pop.clone();

                        for ind in &mut new_pop {
                            let routes = mapping.apply(ind);
                            ind.objectives = acc_qe.evaluate_ind(&routes);
                        }

                        print_population_objectives(
                            &out_folder.join("ConstantModel"),
                            format!("{}_{}.objs", services.len(), evaluations),
                            &new_pop,
                        )
                        .unwrap();

                        model_timer.start();
                    },
                );

                model_timer.pause();

                let mut file = get_file(&out_folder.join("ConstantModel"), "running_time.out");

                write!(file, "{}", model_timer.read()).unwrap();

                // Utilisation Latency Model
                println!("{} {} UtilLatency", topology, scale);

                let util_latency_eval = UtilisationLatencyEval::new(
                    &dc,
                    &rt,
                    &dm,
                    capacities.clone(),
                    &services,
                    &node_selection,
                    acc_qm.clone(),
                );

                let mut model_timer = Stopwatch::new();
                model_timer.start();

                ppls_simple::run(
                    &init_pop,
                    &mapping,
                    &util_latency_eval,
                    neighbour_gen.clone(),
                    128,
                    max_evaluations,
                    2,
                    false,
                    |evaluations, pop| {
                        model_timer.pause();

                        let mut new_pop = pop.clone();

                        for ind in &mut new_pop {
                            let routes = mapping.apply(ind);
                            ind.objectives = acc_qe.evaluate_ind(&routes);
                        }

                        print_population_objectives(
                            &out_folder.join("UtilLatencyModel"),
                            format!("{}_{}.objs", services.len(), evaluations),
                            &new_pop,
                        )
                        .unwrap();

                        model_timer.start();
                    },
                );

                model_timer.pause();

                let mut file = get_file(&out_folder.join("UtilLatencyModel"), "running_time.out");

                write!(file, "{}", model_timer.read()).unwrap();
            }
        }
    }
}

fn run_vlg_tests(args: &Args) {
    println!("Very Large Graph Tests");

    let output_folder = args.output_folder.clone();
    let topologies = args.topologies.clone();
    let max_evaluations = args.max_evaluations;

    let accuracy = args.model.accuracy;
    let converged_iterations = args.model.converged_iterations;

    let test_folder = output_folder.join("VLG");

    for topology in topologies.iter() {
        for scale in &args.test_sizes {
            let dc_rt = load_topology(topology, *scale);
            if dc_rt.is_none() {
                continue;
            }
            let (dc, rt) = dc_rt.unwrap();

            let num_servers = dc.num_servers;

            let num_nearest = (num_servers as f64) * args.perc_nearest;
            let dm = build_cache(&dc, num_nearest.ceil() as usize);

            // Common parameters
            let capacities = vec![100; dc.num_servers];

            for problem_instance in args.run_range() {
                let out_folder = test_folder
                    .join(topology.to_string())
                    .join(scale.to_string())
                    .join(problem_instance.to_string());

                let services = create_problem_instance(args, args.problem.utilisation, num_servers);
                let vnfs = services.iter().map(|s| s).collect();

                let mapping = get_default_mapping(&args, &capacities, &dm, &rt);
                let neighbour_gen = AddRemSwap::new(vnfs);

                let init_pop = get_default_initialisation(&services, &dc);

                // Initialize model
                let sw_sr = args.model.port_sr * dc.num_ports as f64;
                let sw_ql = args.model.port_ql * dc.num_ports;
                let active_cost = args.model.active_cost;
                let idle_cost = args.model.idle_cost;
                let node_selection = args.node_selection.clone();

                let acc_qm = IterativeQueueingModel::new(
                    &dc,
                    sw_sr,
                    sw_ql,
                    accuracy,
                    converged_iterations,
                    active_cost,
                    idle_cost,
                );
                let acc_qe = QueueingEval::new(
                    acc_qm.clone(),
                    &rt,
                    &dm,
                    &capacities,
                    &services,
                    node_selection.clone(),
                );

                let qm_base = IterativeQueueingModel::new(
                    &dc,
                    sw_sr,
                    sw_ql,
                    std::f64::INFINITY,
                    1,
                    active_cost,
                    idle_cost,
                );
                let ue_eval = UtilisationEval::new(
                    &dc,
                    &rt,
                    &dm,
                    capacities.clone(),
                    &services,
                    sw_sr,
                    sw_ql,
                    &node_selection,
                    qm_base,
                );

                let mut run_timer = Stopwatch::new();
                run_timer.start();

                ppls_simple::run(
                    &init_pop,
                    &mapping,
                    &ue_eval,
                    neighbour_gen.clone(),
                    128,
                    max_evaluations,
                    2,
                    false,
                    |evaluations, pop| {
                        run_timer.pause();

                        let mut new_pop = pop.clone();

                        for ind in &mut new_pop {
                            let routes = mapping.apply(ind);
                            ind.objectives = acc_qe.evaluate_ind(&routes);
                        }

                        print_population_objectives(
                            &out_folder,
                            format!("{}_{}.objs", services.len(), evaluations),
                            &new_pop,
                        )
                        .unwrap();

                        run_timer.start();
                    },
                );

                run_timer.pause();

                let mut file = get_file(&out_folder, "running_time.out");

                write!(file, "{}", run_timer.read()).unwrap();
            }
        }
    }
}

// ----- Helper functions -----
fn get_default_initialisation<'a>(
    services: &'a Vec<Service>,
    dc: &Datacentre,
) -> ImprovedServiceAwareInitialisation<'a> {
    ImprovedServiceAwareInitialisation::new(services, dc.num_servers)
}

fn get_default_model<'a>(
    args: &Args,
    services: &'a Vec<Service>,
    dc: &'a Datacentre,
    capacities: &'a Vec<usize>,
    rt: &'a Vec<RoutingTable>,
    dm: &'a DistanceMatrix,
) -> QueueingEval<'a, FirstFit> {
    let sw_sr = args.model.port_sr * dc.num_ports as f64;
    let sw_ql = args.model.port_ql * dc.num_ports;
    let accuracy = args.model.accuracy;
    let converged_iterations = args.model.converged_iterations;
    let active_cost = args.model.active_cost;
    let idle_cost = args.model.idle_cost;
    let node_selection = args.node_selection.clone();

    let qm = IterativeQueueingModel::new(
        &dc,
        sw_sr,
        sw_ql,
        accuracy,
        converged_iterations,
        active_cost,
        idle_cost,
    );

    QueueingEval::new(qm, rt, dm, &capacities, services, node_selection)
}

pub fn get_default_mapping<'a>(
    args: &Args,
    capacities: &'a Vec<usize>,
    dm: &'a DistanceMatrix,
    rt: &'a Vec<RoutingTable>,
) -> ServiceToRouteMapping<'a, FirstFit> {
    ServiceToRouteMapping::new(args.node_selection.clone(), capacities, dm, rt)
}

fn create_problem_instance(args: &Args, utilisation: f64, num_servers: usize) -> Vec<Service> {
    // Problem parameter settings
    let mean_service_len = args.problem.mean_service_len as f64;
    let variance_service_len = args.problem.variance_service_len as f64;
    let max_service_len = args.problem.max_service_len as f64;
    let min_service_len = args.problem.min_service_len as f64;

    let mean_prod_rate = args.problem.mean_prod_rate;
    let variance_prod_rate = args.problem.variance_prod_rate;
    let min_prod_rate = args.problem.min_prod_rate;

    let mean_service_rate = args.problem.mean_service_rate;
    let variance_service_rate = args.problem.variance_service_rate;
    let min_service_rate = args.problem.min_service_rate;

    let queue_length = args.problem.vnf_queue_length;

    let mean_size = args.problem.mean_vnf_size;
    let variance_size = args.problem.variance_vnf_size;

    // Distributions
    let service_len_distr = Normal::new(mean_service_len, variance_service_len).unwrap();
    let prod_rate_distr = Normal::new(mean_prod_rate, variance_prod_rate).unwrap();
    let service_rate_distr = Normal::new(mean_service_rate, variance_service_rate).unwrap();
    let size_distr = Normal::new(mean_size, variance_size).unwrap();

    let num_services =
        (utilisation * (1.0 / mean_service_len) * num_servers as f64).max(1.0) as usize;

    let mut rng = thread_rng();

    loop {
        let mut vnf_id = 0;
        let mut services = Vec::new();
        for service_id in 0..num_services {
            let prod_rate: f64 = prod_rate_distr.sample(&mut rng);
            let prod_rate = prod_rate.max(min_prod_rate);

            let mut service = Service {
                id: service_id,
                prod_rate: prod_rate,
                vnfs: Vec::new(),
            };

            let num_vnfs = service_len_distr
                .sample(&mut rng)
                .max(min_service_len)
                .min(max_service_len);
            let num_vnfs = num_vnfs as usize;

            for _ in 0..num_vnfs {
                let service_rate: f64 = service_rate_distr.sample(&mut rng);
                let service_rate = service_rate.max(min_service_rate);

                let size: f64 = size_distr.sample(&mut rng);
                let size = size.min(100.0).max(1.0) as usize;

                service.vnfs.push(VNF {
                    service_rate,
                    queue_length,
                    size,
                });

                vnf_id = vnf_id + 1;
            }

            services.push(service);
        }

        let mut used_capacity = 0;
        for service in &services {
            for vnf in &service.vnfs {
                used_capacity = used_capacity + vnf.size;
            }
        }

        // Filter out some unsolveable problems
        let total_capacity = 100 * num_servers;
        if used_capacity <= total_capacity {
            return services;
        }
    }
}

fn write_topology(
    name: Topology,
    base_folder: &PathBuf,
    size: usize,
    dc: &Datacentre,
    routing_tables: &Vec<RoutingTable>,
) {
    let out_folder = base_folder.join(format!("{}_{}.dat", name, size));
    let file = File::create(out_folder).unwrap();
    let mut buf_writer = BufWriter::new(file);
    bincode::serialize_into(&mut buf_writer, dc).unwrap();

    let out_folder = base_folder.join(format!("{}_routing_{}.dat", name, size));
    let file = File::create(out_folder).unwrap();
    let mut buf_writer = BufWriter::new(file);
    bincode::serialize_into(&mut buf_writer, routing_tables).unwrap();
}

fn load_topology(topology: &Topology, size: usize) -> Option<(Datacentre, Vec<RoutingTable>)> {
    let file = File::open(format!("topology/compressed/{}_{}.dat", topology, size));
    if file.is_err() {
        return None;
    }
    let file = file.unwrap();

    let reader = BufReader::new(file);
    let dc: Datacentre = bincode::deserialize_from(reader).unwrap();

    let file = File::open(format!(
        "topology/compressed/{}_routing_{}.dat",
        topology, size
    ))
    .unwrap();
    let reader = BufReader::new(file);
    let rt: Vec<RoutingTable> = bincode::deserialize_from(reader).unwrap();

    Some((dc, rt))
}

fn append_to_file(path: &PathBuf, file_name: String, data: &str) {
    fs::create_dir_all(path).unwrap();

    let file = OpenOptions::new()
        .append(true)
        .create(true)
        .open(path.join(file_name));

    if file.is_err() {
        panic!("{}", file.unwrap_err());
    }

    let file = file.unwrap();

    let mut buf_writer = BufWriter::new(file);
    buf_writer.write_all(data.as_bytes()).unwrap()
}

fn print_population_objectives(
    folder: &PathBuf,
    file_name: String,
    pop: &Vec<Solution<Vec<&Service>>>,
) -> std::io::Result<()> {
    print_objectives(
        folder,
        file_name,
        &pop.iter().cloned().map(|ind| ind.objectives).collect(),
    )?;

    Ok(())
}

fn print_objectives(
    folder: &PathBuf,
    file_name: String,
    all_objectives: &Vec<Constraint<Vec<f64>, usize>>,
) -> std::io::Result<()> {
    let mut file = get_file(&folder, &file_name);

    for objectives in all_objectives {
        if objectives.is_feasible() {
            let objectives = objectives.unwrap();

            for (i, objective) in objectives.iter().enumerate() {
                write!(file, "{}", objective)?;

                if i < objectives.len() - 1 {
                    write!(file, ",")?;
                }
            }
        } else {
            write!(file, "Infeasible")?;
        }

        writeln!(file)?;
    }

    Ok(())
}

fn get_file(folder: &PathBuf, file: &str) -> BufWriter<File> {
    fs::create_dir_all(folder).unwrap();
    let path = folder.join(file);

    let file = OpenOptions::new().write(true).create(true).open(path);

    if file.is_err() {
        panic!("{}", file.unwrap_err());
    }

    let file = file.unwrap();
    BufWriter::new(file)
}
