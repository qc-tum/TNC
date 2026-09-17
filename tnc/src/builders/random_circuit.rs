use itertools::Itertools;
use rand::distr::Bernoulli;
use rand::seq::IndexedRandom;
use rand::{Rng, RngExt};

use crate::builders::circuit_builder::{Circuit, Qubit};
use crate::builders::connectivity::{Connectivity, ConnectivityLayout};
use crate::tensornetwork::tensordata::TensorData;

macro_rules! fsim {
    ($a:expr, $b:expr, $c:expr) => {
        $crate::tensornetwork::tensordata::TensorData::Gate((
            String::from("fsim"),
            vec![$a, $b],
            $c,
        ))
    };
}

/// Creates a random circuit.
///
/// The circuit has `rounds` many rounds of single and two qubit gate layers. Gates
/// are placed with the given probabilities and only on qubit pairs specified by the
/// `connectivity`.
pub fn random_circuit<R>(
    qubits: usize,
    rounds: usize,
    single_qubit_probability: f64,
    two_qubit_probability: f64,
    rng: &mut R,
    connectivity: ConnectivityLayout,
) -> Circuit
where
    R: Rng,
{
    let single_qubit_gates = [
        TensorData::Gate((String::from("sx"), Vec::new(), false)),
        TensorData::Gate((String::from("sy"), Vec::new(), false)),
        TensorData::Gate((String::from("sz"), Vec::new(), false)),
    ];

    let single_qubit_die = Bernoulli::new(single_qubit_probability).unwrap();
    let two_qubit_die = Bernoulli::new(two_qubit_probability).unwrap();

    // Get connectivity for given size
    let connectivity_graph = Connectivity::new(connectivity);
    let filtered_connectivity = connectivity_graph
        .connectivity
        .iter()
        .filter(|&&(u, v)| u < qubits && v < qubits)
        .collect_vec();

    // Initialize circuit with random qubit states
    let mut circuit = Circuit::default();
    let qr = circuit.allocate_register("q", qubits);

    for _ in 1..rounds {
        for i in 0..qubits {
            // Placing of random single qubit gate
            if rng.sample(single_qubit_die) {
                let gate = single_qubit_gates.choose(rng).unwrap().clone();
                circuit.append_gate(gate, &[qr.qubit(i)]);
            }
        }
        for (i, j) in &filtered_connectivity {
            // Placing of random two qubit gate
            if rng.sample(two_qubit_die) {
                let gate = fsim!(0.3, 0.2, false);
                circuit.append_gate(gate, &[qr.qubit(*i), qr.qubit(*j)]);
            }
        }
    }

    circuit
}

/// Creates a random observable from X, Y, and Z gates. Can be used together with
/// [`random_circuit`] to build a random expectation value circuit.
pub fn random_observable<R>(
    circuit: &Circuit,
    observable_probability: f64,
    rng: &mut R,
) -> Vec<(TensorData, Vec<Qubit>)>
where
    R: Rng,
{
    let affected_qubits: Vec<_> = circuit
        .qubits()
        .filter(|_| rng.random_bool(observable_probability))
        .collect();

    let observables = [
        TensorData::Gate((String::from("x"), Vec::new(), false)),
        TensorData::Gate((String::from("y"), Vec::new(), false)),
        TensorData::Gate((String::from("z"), Vec::new(), false)),
    ];

    let observable = observables.choose_iter(rng).unwrap().cloned();

    observable
        .zip(affected_qubits)
        .map(|(gate, qubit)| (gate, vec![qubit]))
        .collect()
}
