use tnc::contractionpath::contraction_tree::import::logs_to_pdf;

static LOGGING_FOLDER: &str = "logs/run";

fn main() {
    let filename = format!("{LOGGING_FOLDER}/balanced_usage");
    let suffix = "BestWorst_Greedy.log.json";
    let ranks = 4;
    logs_to_pdf(
        &filename,
        suffix,
        ranks,
        &format!("{LOGGING_FOLDER}/contraction_BestWorst_Greedy"),
    );

    let suffix = "Tensor_Greedy.log.json";
    let ranks = 4;
    logs_to_pdf(
        &filename,
        suffix,
        ranks,
        &format!("{LOGGING_FOLDER}/contraction_Tensor_Greedy"),
    );

    let suffix = "Tensors_Greedy.log.json";
    let ranks = 4;
    logs_to_pdf(
        &filename,
        suffix,
        ranks,
        &format!("{LOGGING_FOLDER}/contraction_Tensors_Greedy"),
    );

    let suffix = "Unoptimized.log.json";
    let ranks = 4;
    logs_to_pdf(
        &filename,
        suffix,
        ranks,
        &format!("{LOGGING_FOLDER}/contraction_Unoptimized"),
    );
}
