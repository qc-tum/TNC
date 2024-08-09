use std::ffi::CString;

pub enum PartitioningStrategy {
    MinCut,
    CommunityFinding,
    File(String),
}

pub(super) fn to_c_string(strategy: PartitioningStrategy) -> CString {
    match strategy {
        PartitioningStrategy::MinCut => {
            CString::new(include_str!("./cut_kKaHyPar_sea20.ini")).expect("Unable to parse file")
        }
        PartitioningStrategy::CommunityFinding => {
            CString::new(include_str!("./cut_kKaHyPar_sea20.ini")).expect("Unable to parse file")
        }
        PartitioningStrategy::File(str) => CString::new(str).expect("Unable to parse string"),
    }
}
