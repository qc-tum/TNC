use tensorcontraction::contractionpath::contraction_tree::import::logs_to_pdf;

fn main() {
    let filename = String::from("balanced_usage");
    let suffix = String::from("opt.log.json");
    let ranks = 4;
    logs_to_pdf(filename, suffix, ranks, String::from("opt"));

    let filename = String::from("balanced_usage");
    let suffix = String::from("unopt.log.json");
    let ranks = 4;
    logs_to_pdf(filename, suffix, ranks, String::from("unopt"));
}
