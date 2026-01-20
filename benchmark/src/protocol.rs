use std::{fs, io::Write};

use serde::{Deserialize, Serialize};

/// A log entry documents the progress of a single run.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
enum LogEntry {
    /// The run with the given ID is currently attempted.
    Trying(usize),
    /// The run with the given ID has been completed.
    Done(usize),
    /// The run with the given ID has failed.
    Error(usize),
}

/// A protocol is a list of [`LogEntry`]s, describing all runs that have been attempted.
#[derive(Clone, Default, Debug, Serialize, Deserialize)]
pub struct Protocol(Vec<LogEntry>);

impl Protocol {
    /// Reads a protocol from a file. Converts all `Trying` entries to `Error`, because
    /// they failed to finish.
    pub fn from_file(file: fs::File) -> Self {
        let mut protocol: Self = serde_json::from_reader(file).unwrap();
        protocol.0.iter_mut().for_each(|entry| {
            if let LogEntry::Trying(id) = entry {
                *entry = LogEntry::Error(*id);
            }
        });
        protocol
    }

    /// Writes the protocol to a file.
    fn write(&self, filename: &str) {
        let mut file = fs::File::create(filename).unwrap();
        serde_json::to_writer(&mut file, &self).unwrap();
        file.flush().unwrap();
    }

    /// Checks if the protocol contains a run with the given ID (Done or Error).
    pub fn contains(&self, id: usize) -> bool {
        self.0.iter().any(|entry| match entry {
            LogEntry::Done(x) | LogEntry::Error(x) => *x == id,
            LogEntry::Trying(_) => panic!("Trying entry should not be in the protocol"),
        })
    }

    /// Writes a `Trying` entry to the protocol and saves it to file.
    pub fn write_trying(&mut self, id: usize) {
        self.0.push(LogEntry::Trying(id));
        self.write("protocol.json");
    }

    /// Writes a `Done` entry to the protocol and saves it to file.
    pub fn write_done(&mut self, id: usize) {
        let LogEntry::Trying(x) = self.0.pop().unwrap() else {
            panic!("Expected a trying entry when writing a done entry")
        };
        assert_eq!(id, x);
        self.0.push(LogEntry::Done(id));
        self.write("protocol.json");
    }
}
