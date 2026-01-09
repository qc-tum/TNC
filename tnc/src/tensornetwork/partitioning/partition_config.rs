use std::{
    ffi::{CStr, CString},
    path::PathBuf,
};

use kahypar::{include_cstr, KaHyParContext};

static MIN_CUT_CONFIG: &CStr = include_cstr!("cut_kKaHyPar_sea20.ini");
static COMMUNITY_FINDING_CONFIG: &CStr = include_cstr!("km1_kKaHyPar_sea20.ini");

/// Different strategies for partitioning a tensor network.
pub enum PartitioningStrategy {
    /// Uses the min cut heuristic.
    MinCut,
    /// Uses the community finding heuristic.
    CommunityFinding,
    /// A custom `KaHyPar` configuration loaded from an INI file.
    Custom(PathBuf),
}

impl PartitioningStrategy {
    pub(super) fn apply(self, context: &mut KaHyParContext) {
        match self {
            PartitioningStrategy::MinCut => {
                context.configure_from_str(MIN_CUT_CONFIG);
            }
            PartitioningStrategy::CommunityFinding => {
                context.configure_from_str(COMMUNITY_FINDING_CONFIG);
            }
            PartitioningStrategy::Custom(path) => {
                context
                    .configure_from_file(CString::new(path.to_str().unwrap()).unwrap().as_c_str());
            }
        }
    }
}
