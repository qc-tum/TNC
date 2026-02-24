//! Generating circuits similar to the Sycamore circuits.

use std::f64::consts::{FRAC_PI_2, FRAC_PI_6};

use rand::{seq::IndexedRandom, Rng};

use crate::{
    builders::{
        circuit_builder::Circuit,
        connectivity::{sycamore_a, sycamore_b, sycamore_c, sycamore_d},
    },
    tensornetwork::tensordata::TensorData,
};

/// Creates a circuit based on the Sycamore circuit scheme.
///
/// The `depth` is the number of rounds, where one round consists of a layer of
/// single-qubit gates followed by a layer of two-qubit gates. The circuit is
/// initialized in the |0> state.
///
/// For more details on the circuit, see <https://arxiv.org/abs/1910.11333>.
pub fn sycamore_circuit<R>(qubits: usize, depth: usize, rng: &mut R) -> Circuit
where
    R: Rng,
{
    // TODO: if we generalize the connection patterns, we can allow arbitrary sizes here
    assert!(
        qubits <= 49,
        "Currently only supports circuits of size equal to the original Sycamore experiment"
    );
    let mut rounds = [
        sycamore_a, sycamore_b, sycamore_c, sycamore_d, sycamore_c, sycamore_d, sycamore_a,
        sycamore_b,
    ]
    .iter()
    .cycle();
    let single_qubit_gates = [
        TensorData::Gate((String::from("sx"), Vec::new(), false)),
        TensorData::Gate((String::from("sy"), Vec::new(), false)),
        TensorData::Gate((String::from("sz"), Vec::new(), false)),
    ];
    let two_qubit_gate =
        TensorData::Gate((String::from("fsim"), vec![FRAC_PI_2, FRAC_PI_6], false));

    // Initialize circuit
    let mut circuit = Circuit::default();
    let qreg = circuit.allocate_register(qubits);

    // Add interleaved layers of random single-qubit gates and two-qubit gates
    for round in 0..=depth {
        // Add single-qubit gate layer
        for i in 0..qubits {
            let gate = single_qubit_gates.choose(rng).unwrap().clone();
            circuit.append_gate(gate, &[qreg.qubit(i)]);
        }

        // In the last round, we only have a final single-qubit layer.
        if round < depth {
            // Add two-qubit gates with round-specific connectivity.
            let layer = rounds.next().unwrap()();
            for (i, j) in layer {
                if i > qubits || j > qubits {
                    continue;
                }
                let i = i - 1;
                let j = j - 1;
                circuit.append_gate(two_qubit_gate.clone(), &[qreg.qubit(i), qreg.qubit(j)]);
            }
        }
    }
    circuit
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use rand::{rngs::StdRng, SeedableRng};

    use super::*;

    #[test]
    fn small_sycamore() {
        let mut rng = StdRng::seed_from_u64(42);
        let circuit = sycamore_circuit(3, 3, &mut rng);
        let (tn, _) = circuit.into_amplitude_network(&"0".repeat(3));

        let rank_counts = tn.tensors().iter().counts_by(|t| t.legs().len());
        assert_eq!(rank_counts.len(), 3);
        // 3 initial state, 3 final state
        assert_eq!(rank_counts[&1], 6);
        // 4 * 3 single qubit gates
        assert_eq!(rank_counts[&2], 12);
        // 1 two-qubit gate (in round C)
        assert_eq!(rank_counts[&4], 1);
    }
}
