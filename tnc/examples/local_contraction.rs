//! This example shows how to load a QASM2 program, create a tensor network that
//! computes the state vector out of it and contract this tensor network locally.

use tnc::{
    contractionpath::paths::{
        cotengrust::{Cotengrust, OptMethod},
        OptimizePath,
    },
    qasm::import_qasm,
    tensornetwork::contraction::contract_tensor_network,
};

fn main() {
    // The QASM code prepares a GHZ state
    let code = "\
OPENQASM 2.0;
include \"qelib1.inc\";

qreg q[3];
h q[0];
cx q[0], q[1];
cx q[1], q[2];
";

    // Create a Circuit instance out of the code
    let circuit = import_qasm(code);

    // Create a tensor network that computes the full state vector.
    // This also returns a permutator which we need to apply to the final tensor to
    // make sure the entries are in the expected order.
    let (tensor_network, permutator) = circuit.into_statevector_network();

    // Find a contraction path to contract the tensor network.
    // We use a greedy path finder here.
    let mut opt = Cotengrust::new(&tensor_network, OptMethod::Greedy);
    opt.optimize_path();
    let path = opt.get_best_replace_path();

    // Contract the tensor network locally
    let final_tensor = contract_tensor_network(tensor_network, &path);

    // Apply the permutator to make sure the data is in the expected order
    let statevector = permutator.apply(final_tensor);

    // Get the data vector. Don't worry, the clone does not clone the data itself.
    let data = statevector.tensor_data().clone().into_data();

    // Print the data
    println!("Resulting statevector is: {:?}", data.elements());
}
