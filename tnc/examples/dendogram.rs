use rand::{rngs::StdRng, SeedableRng};
use tnc::{
    builders::{connectivity::ConnectivityLayout, random_circuit::random_circuit},
    contractionpath::{
        contraction_cost::contract_cost_tensors,
        contraction_tree::{
            export::{to_dendogram_format, to_pdf},
            ContractionTree,
        },
        paths::{
            cotengrust::{Cotengrust, OptMethod},
            OptimizePath,
        },
    },
    tensornetwork::partitioning::{
        find_partitioning, partition_config::PartitioningStrategy, partition_tensor_network,
    },
};

fn main() {
    let mut rng = StdRng::seed_from_u64(23);
    let tensor = random_circuit(5, 10, 0.4, 0.4, &mut rng, ConnectivityLayout::Osprey);
    let partitioning = find_partitioning(&tensor, 3, PartitioningStrategy::MinCut, true);

    let partitioned_tn = partition_tensor_network(tensor, &partitioning);
    let mut opt = Cotengrust::new(&partitioned_tn, OptMethod::Greedy);

    opt.optimize_path();
    let path = opt.get_best_replace_path();

    let contraction_tree = ContractionTree::from_contraction_path(&partitioned_tn, &path);

    let dendogram_entries =
        to_dendogram_format(&contraction_tree, &partitioned_tn, contract_cost_tensors);
    to_pdf("new_method", &dendogram_entries, None);
}
