use lazy_static::lazy_static;
use num_complex::Complex64;
use permutation::Permutation;
use tetra::Tensor as DataTensor;

use std::{
    borrow::Borrow,
    collections::HashSet,
    f64::consts::FRAC_1_SQRT_2,
    hash::{Hash, Hasher},
    sync::RwLock,
};

lazy_static! {
    static ref GATES: RwLock<HashSet<Box<dyn Gate>>> = {
        let mut gates = HashSet::new();
        gates.insert(Box::new(X) as _);
        gates.insert(Box::new(Y) as _);
        gates.insert(Box::new(Z) as _);
        gates.insert(Box::new(H) as _);
        gates.insert(Box::new(U) as _);
        gates.insert(Box::new(Sx) as _);
        gates.insert(Box::new(Sy) as _);
        gates.insert(Box::new(Sz) as _);
        gates.insert(Box::new(Cx) as _);
        gates.insert(Box::new(Fsim) as _);
        RwLock::new(gates)
    };
}

/// Registers a gate definition to resolve a gate name to a gate implementation.
pub fn register_gate(gate: Box<dyn Gate>) {
    assert!(
        gate.name().to_ascii_lowercase() == gate.name(),
        "Gate name must be lowercase."
    );
    GATES.write().unwrap().insert(gate);
}

/// Computes the gate matrix for the given gate and angles.
#[must_use]
pub fn load_gate(gate: &str, angles: &[f64]) -> DataTensor {
    let gates = &GATES.read().unwrap();
    let gate = gates
        .get(gate)
        .unwrap_or_else(|| panic!("Gate '{}' not found.", gate));
    gate.compute(angles)
}

/// Returns whether the given gate is known.
#[must_use]
pub fn is_gate_known(gate: &str) -> bool {
    let gates = &GATES.read().unwrap();
    gates.contains(gate)
}

/// Helper method to compute the adjoint (conjugate transpose) of a data tensor
/// in-place. The data tensor can be a matrix, or also be split in dimensions of
/// size 2. For example, both `(8,8)` or `(2,2,2,2,2,2)` are okay.
fn apply_adjoint(data: &mut DataTensor) {
    if data.ndim() > 0 {
        assert!(data.ndim().is_power_of_two());
        let half = data.ndim() / 2;
        let perm = (half..data.ndim()).chain(0..half).collect::<Vec<_>>();
        data.transpose(&Permutation::oneline(perm));
    }
    data.conjugate();
}

/// A quantum gate.
pub trait Gate: Send + Sync {
    /// Returns the name of the gate.
    fn name(&self) -> &str;

    /// Computes the gate matrix with the given angles.
    fn compute(&self, angles: &[f64]) -> DataTensor;

    /// Computes the adjoint of the gate matrix with the given angles. If not
    /// overridden, this computes the conjugate transpose of the gate matrix.
    fn adjoint(&self, angles: &[f64]) -> DataTensor {
        let mut matrix = self.compute(angles);
        apply_adjoint(&mut matrix);
        matrix
    }
}

impl PartialEq for dyn Gate {
    fn eq(&self, other: &Self) -> bool {
        self.name() == other.name()
    }
}

impl Eq for dyn Gate {}

impl Hash for dyn Gate {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.name().hash(state);
    }
}

/// This allows us to use a `&str` as a key in a `HashSet` of gates.
impl Borrow<str> for Box<dyn Gate> {
    fn borrow(&self) -> &str {
        self.name()
    }
}

/// The Pauli-X gate.
struct X;
impl Gate for X {
    fn name(&self) -> &str {
        "x"
    }

    fn compute(&self, angles: &[f64]) -> DataTensor {
        assert!(angles.is_empty());
        let z = Complex64::ZERO;
        let o = Complex64::ONE;
        #[rustfmt::skip]
        let data = vec![
            z, o,
            o, z,
        ];
        DataTensor::new_from_flat(&[2, 2], data, None)
    }

    fn adjoint(&self, angles: &[f64]) -> DataTensor {
        // self-adjoint
        self.compute(angles)
    }
}

/// The Pauli-Y gate.
struct Y;
impl Gate for Y {
    fn name(&self) -> &str {
        "y"
    }

    fn compute(&self, angles: &[f64]) -> DataTensor {
        assert!(angles.is_empty());
        let z = Complex64::ZERO;
        let i = Complex64::I;
        #[rustfmt::skip]
        let data = vec![
            z, -i,
            i,  z,
        ];
        DataTensor::new_from_flat(&[2, 2], data, None)
    }

    fn adjoint(&self, angles: &[f64]) -> DataTensor {
        // self-adjoint
        self.compute(angles)
    }
}

/// The Pauli-Z gate.
struct Z;
impl Gate for Z {
    fn name(&self) -> &str {
        "z"
    }

    fn compute(&self, angles: &[f64]) -> DataTensor {
        assert!(angles.is_empty());
        let z = Complex64::ZERO;
        let o = Complex64::ONE;
        #[rustfmt::skip]
        let data = vec![
            o,  z,
            z, -o,
        ];
        DataTensor::new_from_flat(&[2, 2], data, None)
    }

    fn adjoint(&self, angles: &[f64]) -> DataTensor {
        // self-adjoint
        self.compute(angles)
    }
}

/// The Hadamard gate.
struct H;
impl Gate for H {
    fn name(&self) -> &str {
        "h"
    }

    fn compute(&self, angles: &[f64]) -> DataTensor {
        assert!(angles.is_empty());
        let h = Complex64::new(FRAC_1_SQRT_2, 0.0);
        #[rustfmt::skip]
        let data = vec![
            h,  h,
            h, -h,
        ];
        DataTensor::new_from_flat(&[2, 2], data, None)
    }

    fn adjoint(&self, angles: &[f64]) -> DataTensor {
        // self-adjoint
        self.compute(angles)
    }
}

/// The U gate with three parameters, following the [OpenQASM 3.0 specification](https://openqasm.com/language/gates.html#built-in-gates).
struct U;
impl Gate for U {
    fn name(&self) -> &str {
        "u"
    }

    fn compute(&self, angles: &[f64]) -> DataTensor {
        let [theta, phi, lambda] = angles else {
            panic!("Expected 3 angles, got {}", angles.len())
        };
        let (sin, cos) = (theta / 2.0).sin_cos();
        let data = vec![
            Complex64::new(cos, 0.0),
            -(Complex64::I * lambda).exp() * sin,
            (Complex64::I * phi).exp() * sin,
            (Complex64::I * (phi + lambda)).exp() * cos,
        ];
        DataTensor::new_from_flat(&[2, 2], data, None)
    }
}

/// The square-root of X gate.
struct Sx;
impl Gate for Sx {
    fn name(&self) -> &str {
        "sx"
    }

    fn compute(&self, angles: &[f64]) -> DataTensor {
        assert!(angles.is_empty());
        let a = Complex64::new(0.5, 0.5);
        let b = Complex64::new(0.5, -0.5);
        #[rustfmt::skip]
        let data = vec![
            a, b,
            b, a,
        ];
        DataTensor::new_from_flat(&[2, 2], data, None)
    }

    fn adjoint(&self, angles: &[f64]) -> DataTensor {
        assert!(angles.is_empty());
        let a = Complex64::new(0.5, -0.5);
        let b = Complex64::new(0.5, 0.5);
        #[rustfmt::skip]
        let data = vec![
            a, b,
            b, a,
        ];
        DataTensor::new_from_flat(&[2, 2], data, None)
    }
}

/// The square-root of Y gate.
struct Sy;
impl Gate for Sy {
    fn name(&self) -> &str {
        "sy"
    }

    fn compute(&self, angles: &[f64]) -> DataTensor {
        assert!(angles.is_empty());
        let a = Complex64::new(0.5, 0.5);
        let b = Complex64::new(-0.5, -0.5);
        #[rustfmt::skip]
        let data = vec![
            a, b,
            a, a,
        ];
        DataTensor::new_from_flat(&[2, 2], data, None)
    }

    fn adjoint(&self, angles: &[f64]) -> DataTensor {
        assert!(angles.is_empty());
        let a = Complex64::new(0.5, -0.5);
        let b = Complex64::new(-0.5, 0.5);
        #[rustfmt::skip]
        let data = vec![
            a, a,
            b, a,
        ];
        DataTensor::new_from_flat(&[2, 2], data, None)
    }
}

/// The square-root of Z gate.
struct Sz;
impl Gate for Sz {
    fn name(&self) -> &str {
        "sz"
    }

    fn compute(&self, angles: &[f64]) -> DataTensor {
        assert!(angles.is_empty());
        let z = Complex64::ZERO;
        let o = Complex64::ONE;
        let i = Complex64::I;
        #[rustfmt::skip]
        let data = vec![
            o, z,
            z, i,
        ];
        DataTensor::new_from_flat(&[2, 2], data, None)
    }

    fn adjoint(&self, angles: &[f64]) -> DataTensor {
        assert!(angles.is_empty());
        let z = Complex64::ZERO;
        let o = Complex64::ONE;
        let i = Complex64::I;
        #[rustfmt::skip]
        let data = vec![
            o,  z,
            z, -i,
        ];
        DataTensor::new_from_flat(&[2, 2], data, None)
    }
}

/// The controlled-X gate.
struct Cx;
impl Gate for Cx {
    fn name(&self) -> &str {
        "cx"
    }

    fn compute(&self, angles: &[f64]) -> DataTensor {
        assert!(angles.is_empty());
        let z = Complex64::ZERO;
        let o = Complex64::ONE;
        #[rustfmt::skip]
        let data = vec![
            o, z, z, z,
            z, o, z, z,
            z, z, z, o,
            z, z, o, z,
        ];
        DataTensor::new_from_flat(&[2, 2, 2, 2], data, None)
    }

    fn adjoint(&self, angles: &[f64]) -> DataTensor {
        // self-adjoint
        self.compute(angles)
    }
}

/// The FSIM gate, as described e.g. [here](https://quantumai.google/reference/python/cirq/FSimGate).
struct Fsim;
impl Gate for Fsim {
    fn name(&self) -> &str {
        "fsim"
    }

    fn compute(&self, angles: &[f64]) -> DataTensor {
        let [theta, phi] = angles else {
            panic!("Expected 2 angles, got {}", angles.len())
        };
        let z = Complex64::ZERO;
        let o = Complex64::ONE;
        let a = Complex64::new(theta.cos(), 0.0);
        let b = Complex64::new(0.0, -theta.sin());
        let c = Complex64::new(0.0, -phi).exp();
        #[rustfmt::skip]
        let data = vec![
            o, z, z, z,
            z, a, b, z,
            z, b, a, z,
            z, z, z, c,
        ];
        DataTensor::new_from_flat(&[2, 2, 2, 2], data, None)
    }
}
