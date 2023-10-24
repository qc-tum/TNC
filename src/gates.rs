use num_complex::Complex64;
use tetra::Tensor as DataTensor;

pub fn load_gate(gate: &'static str) -> DataTensor {
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
        "CX" => DataTensor::new_from_flat(&[4, 4], CX.to_vec(), None),
        _ => todo!(),
    }
}
