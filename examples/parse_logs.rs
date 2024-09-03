use tensorcontraction::contractionpath::contraction_tree::import::logs_to_pdf;

static LOGGING_FOLDER: &str = "logs/run3";

fn main() {
    let filename = format!("{LOGGING_FOLDER}/run3/balanced_usage");
    let suffix = "BestWorst_Greedy.log.json";
    let ranks = 4;
    logs_to_pdf(&filename, suffix, ranks, "contraction_BestWorst_Greedy");

    let suffix = "Tensor_Greedy.log.json";
    let ranks = 4;
    logs_to_pdf(&filename, suffix, ranks, "contraction_Tensor_Greedy");

    let suffix = "Unoptimized.log.json";
    let ranks = 4;
    logs_to_pdf(&filename, suffix, ranks, "contraction_Unoptimized");
}
