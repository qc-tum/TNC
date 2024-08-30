use tensorcontraction::contractionpath::contraction_tree::import::logs_to_pdf;

fn main() {
    let filename = "balanced_usage";
    let suffix = "opt.log.json";
    let ranks = 2;

    logs_to_pdf(filename, suffix, ranks, "contraction_optimized");

    let filename = "balanced_usage";
    let suffix = "unopt.log.json";
    let ranks = 2;

    logs_to_pdf(filename, suffix, ranks, "contraction_unoptimized");
}
