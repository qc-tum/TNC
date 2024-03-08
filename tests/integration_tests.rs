use mpi::traits::{Communicator, CommunicatorCollectives};
use rand::{rngs::StdRng, SeedableRng};
use tensorcontraction::{
    circuits::{connectivity::ConnectivityLayout, sycamore::sycamore_circuit},
    contractionpath::paths::{greedy::Greedy, CostType, OptimizePath},
    mpi::communication::{naive_reduce_tensor_network, scatter_tensor_network},
    tensornetwork::{
        contraction::contract_tensor_network,
        partitioning::{find_partitioning, partition_tensor_network},
        tensor::Tensor,
    },
};

#[test]
fn test_partitioned_contraction() {
    let mut rng = StdRng::seed_from_u64(52);
    let k = 10;

    let r_tn = sycamore_circuit(k, 10, 0.4, 0.4, &mut rng, ConnectivityLayout::Osprey);
    let mut ref_tn = r_tn.clone();
    let mut ref_opt = Greedy::new(&ref_tn, CostType::Flops);
    ref_opt.optimize_path();
    let ref_path = ref_opt.get_best_replace_path();
    contract_tensor_network(&mut ref_tn, &ref_path);

    let partitioning =
        find_partitioning(&r_tn, 3, String::from("tests/km1_kKaHyPar_sea20.ini"), true);
    let mut partitioned_tn = partition_tensor_network(&r_tn, &partitioning);
    let mut opt = Greedy::new(&partitioned_tn, CostType::Flops);
    opt.optimize_path();
    let path = opt.get_best_replace_path();
    contract_tensor_network(&mut partitioned_tn, &path);
    assert!(&ref_tn.approx_eq(&partitioned_tn, 1e-12));
}

/// Given the module path and name of a test function, returns the name as it is
/// used by cargo test.
///
/// # Examples
/// ```ignore
/// assert_eq!(make_full_test_name("mycrate", "my_test"), "my_test");
/// assert_eq!(make_full_test_name("mycrate::foo", "my_test"), "foo::my_test");
/// assert_eq!(make_full_test_name("mycrate::foo::bar", "my_test"), "foo::bar::my_test");
/// ```
pub(crate) fn make_full_test_name(module_path: &str, test_name: &str) -> String {
    if let Some(idx) = module_path.find("::") {
        // Not in the root module, remove the root name and concat test name
        module_path[idx + 2..].to_string() + "::" + test_name
    } else {
        // In the root module, only use test name
        test_name.to_string()
    }
}

/// Runs a test using `mpirun` with 4 processes. The test must be passed with as full
/// name, e.g., `foo::bar::my_test`.
pub(crate) fn run_mpi_test(test_full_name: &str, processes: usize) {
    let mut command = std::process::Command::new("mpirun");
    command
        .arg("-n")
        .arg(processes.to_string())
        .arg("--allow-run-as-root")
        .arg("cargo")
        .arg("test")
        .arg(test_full_name)
        .arg("--")
        .arg("--ignored")
        .arg("--exact");
    let output = command.status().expect("failed to execute process");
    assert!(output.success());
}

#[macro_export]
macro_rules! mpi_test {
    ($processes:expr, fn $name:ident $_:tt $body:block) => {
        paste::paste! {
            #[test]
            fn $name() {
                let full_path = module_path!();
                let test_name = concat!(stringify!($name), "_internal");
                let exact_name = make_full_test_name(full_path, test_name);
                run_mpi_test(&exact_name, $processes);
            }

            #[test]
            #[ignore]
            fn [<$name _internal>]() $body
        }
    };
}

mpi_test!(
    4,
    fn test_partitioned_contraction_need_mpi() {
        let mut rng = StdRng::seed_from_u64(23);

        let universe = mpi::initialize().unwrap();
        let world = universe.world();
        let size = world.size();
        let rank = world.rank();

        // let status;
        let mut partitioned_tn = Tensor::default();
        let mut path = Vec::new();
        let mut ref_tn = Tensor::default();
        if rank == 0 {
            let k = 5;
            let r_tn = sycamore_circuit(k, 10, 0.4, 0.4, &mut rng, ConnectivityLayout::Osprey);
            ref_tn = r_tn.clone();
            let partitioning = find_partitioning(
                &r_tn,
                size,
                String::from("tests/km1_kKaHyPar_sea20.ini"),
                true,
            );
            partitioned_tn = partition_tensor_network(&r_tn, &partitioning);
            let mut opt = Greedy::new(&partitioned_tn, CostType::Flops);

            opt.optimize_path();
            path = opt.get_best_replace_path();
        }
        world.barrier();
        let (mut local_tn, local_path) =
            scatter_tensor_network(partitioned_tn, &path, rank, size, &world);
        contract_tensor_network(&mut local_tn, &local_path);

        naive_reduce_tensor_network(&mut local_tn, &path, rank, size, &world);
        world.barrier();

        if rank == 0 {
            let mut ref_opt = Greedy::new(&ref_tn, CostType::Flops);

            ref_opt.optimize_path();
            let ref_path = ref_opt.get_best_replace_path();
            contract_tensor_network(&mut ref_tn, &ref_path);
            assert!(local_tn.approx_eq(&ref_tn, 1e-8));
        }
    }
);
