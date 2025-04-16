use clap::Parser;

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum Mode {
    Sweep,
    Run,
}

#[derive(Debug, Parser)]
#[command(version, about, long_about=None)]
pub struct Cli {
    #[arg(short, long, value_delimiter = ',', num_args = 1..)]
    pub partitions: Vec<i32>,
    #[arg(short, long, value_delimiter = ',', num_args = 1..)]
    pub include: Vec<String>,
    #[arg(short, long, value_delimiter = ',', num_args = 1..)]
    pub exclude: Vec<String>,
    #[arg(short, long, default_value_t = 1)]
    pub num_seeds: usize,
    pub cache_dir: String,
    pub mode: Mode,
    pub out_file: String,
}
