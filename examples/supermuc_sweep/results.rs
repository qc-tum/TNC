use std::{fs, time::Duration};

use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct OptimizationResult {
    pub num_qubits: usize,
    pub circuit_depth: usize,
    pub seed: u64,
    pub partitions: i32,
    pub actual_partitions: i32,
    pub method: String,
    pub optimization_time: Duration,
    pub flops: f64,
    pub memory: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RunResult {
    pub num_qubits: usize,
    pub circuit_depth: usize,
    pub seed: u64,
    pub partitions: i32,
    pub method: String,
    pub time_to_solution: Duration,
}

#[derive(Default)]
pub struct Writer(Option<fs::File>);

impl Writer {
    pub fn new(filename: &str) -> Self {
        let file = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(filename)
            .unwrap();
        Self(Some(file))
    }

    pub fn write<R>(&mut self, result: R)
    where
        R: Serialize,
    {
        if let Some(file) = &mut self.0 {
            serde_json::to_writer(file, &[result]).unwrap();
        }
    }
}
