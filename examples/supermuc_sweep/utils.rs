use std::{
    collections::HashSet,
    hash::{DefaultHasher, Hash, Hasher},
};

use flexi_logger::{json_format, Duplicate, FileSpec, Logger};
use itertools::Itertools;
use log::LevelFilter;
use mpi::Rank;

/// Sets up logging for rank `rank`. Each rank logs to a separate file and to stdout.
pub fn setup_logging_mpi(rank: Rank) {
    let _logger = Logger::with(LevelFilter::Debug)
        .format(json_format)
        .log_to_file(
            FileSpec::default()
                .discriminant(format!("rank{rank}"))
                .suppress_timestamp()
                .suffix("log.json"),
        )
        .duplicate_to_stdout(Duplicate::Info)
        .start()
        .unwrap();
}

/// Parses a list of numbers and ranges into a set of numbers.
/// E.g. `1,3-4,7-9` -> `{1, 3, 4, 7, 8, 9}`.
pub fn parse_range_list(entries: &[String]) -> HashSet<usize> {
    let mut out = HashSet::new();
    for entry in entries {
        if entry.contains("-") {
            // An inclusive range
            let parts = entry.split("-").collect_vec();
            assert_eq!(parts.len(), 2);
            let start = parts[0].parse().unwrap();
            let end = parts[1].parse().unwrap();
            for i in start..=end {
                out.insert(i);
            }
        } else {
            // A single number
            let number = entry.parse().unwrap();
            out.insert(number);
        }
    }
    out
}

/// Computes the hash of a string.
pub fn hash_str(s: &str) -> u64 {
    let mut hasher = DefaultHasher::new();
    s.hash(&mut hasher);
    hasher.finish()
}
