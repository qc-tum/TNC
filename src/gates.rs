use num_complex::Complex64;
use tetra::Tensor as DataTensor;

use crate::tensornetwork::tensordata::TensorData;

pub fn load_gate(gate: &str, angles: Option<Vec<f64>>) -> DataTensor {
    const X: [Complex64; 4] = [
        Complex64::new(0.0, 0.0),
        Complex64::new(1.0, 0.0),
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
    ];
    const Y: [Complex64; 4] = [
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, -1.0),
        Complex64::new(0.0, 1.0),
        Complex64::new(0.0, 0.0),
    ];

    const Z: [Complex64; 4] = [
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(-1.0, 0.0),
    ];

    const SQRX: [Complex64; 4] = [
        Complex64::new(0.5, 0.5),
        Complex64::new(0.5, -0.5),
        Complex64::new(0.5, -0.5),
        Complex64::new(0.5, 0.5),
    ];
    const SQRY: [Complex64; 4] = [
        Complex64::new(0.5, 0.5),
        Complex64::new(-0.5, -0.5),
        Complex64::new(0.5, 0.5),
        Complex64::new(0.5, 0.5),
    ];

    const SQRZ: [Complex64; 4] = [
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 1.0),
    ];

    const CX: [Complex64; 16] = [
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
    ];

    match gate {
        "X" => DataTensor::new_from_flat(&[2, 2], X.to_vec(), None),
        "Y" => DataTensor::new_from_flat(&[2, 2], Y.to_vec(), None),
        "Z" => DataTensor::new_from_flat(&[2, 2], Z.to_vec(), None),
        "CX" => DataTensor::new_from_flat(&[2, 2, 2, 2], CX.to_vec(), None),
        "SQRX" => DataTensor::new_from_flat(&[2, 2], SQRX.to_vec(), None),
        "SQRY" => DataTensor::new_from_flat(&[2, 2], SQRY.to_vec(), None),
        "SQRZ" => DataTensor::new_from_flat(&[2, 2], SQRZ.to_vec(), None),
        "FSIM" => {
            let mut fsim: [Complex64; 16] = [
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
            ];
            let angles = angles.unwrap();
            fsim[5] = Complex64::new(angles[0].cos(), 0.0);
            fsim[10] = Complex64::new(angles[0].cos(), 0.0);
            fsim[6] = Complex64::new(0.0, angles[0].sin());
            fsim[11] = Complex64::new(0.0, angles[0].sin());
            fsim[15] = Complex64::new(0.0, angles[1].exp());
            DataTensor::new_from_flat(&[2, 2, 2, 2], fsim.to_vec(), None)
        }
        _ => todo!(), // _ => capture_gate(&gate),
    }
}

// fn capture_gate(gate: &str) {
//     let input_re = Regex::new(r#"([a-zA-Z]*)\(((([0-9]*[.])?[0-9]+[\s,]*)+)\)"#).unwrap();
//     let captures = input_re.captures(gate).unwrap();
//     let gate_name = &captures[1];
//     let gate_variables = '[' + &captures[2] + ']';
// }

pub const PAULIX: TensorData = TensorData::Gate(("X", Vec::new()));
pub const PAULIY: TensorData = TensorData::Gate(("Y", Vec::new()));
pub const PAULIZ: TensorData = TensorData::Gate(("Z", Vec::new()));
pub const SQRX: TensorData = TensorData::Gate(("SQRX", Vec::new()));
pub const SQRY: TensorData = TensorData::Gate(("SQRY", Vec::new()));
pub const SQRZ: TensorData = TensorData::Gate(("SQRZ", Vec::new()));
pub const CNOT: TensorData = TensorData::Gate(("CNOT", Vec::new()));

#[macro_export]
macro_rules! fsim {
    ($a:expr, $b:expr) => {
        $crate::tensornetwork::tensordata::TensorData::Gate(("FSIM", vec![$a, $b]))
    };
}
    
