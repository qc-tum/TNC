use std::{ffi::CString, path::PathBuf};

use kahypar_sys::KaHyParContext;

static MIN_CUT_CONFIG: &str = include_str!("cut_kKaHyPar_sea20.ini");
static COMMUNITY_FINDING_CONFIG: &str = include_str!("km1_kKaHyPar_sea20.ini");

pub enum PartitioningStrategy {
    MinCut,
    CommunityFinding,
    /// A custom `KaHyPar` configuration loaded from a INI file.
    Custom(PathBuf),
}

impl PartitioningStrategy {
    pub(super) fn apply(self, context: &mut KaHyParContext) {
        match self {
            PartitioningStrategy::MinCut => {
                context.configure_from_str(CString::new(MIN_CUT_CONFIG).unwrap())
            }
            PartitioningStrategy::CommunityFinding => {
                context.configure_from_str(CString::new(COMMUNITY_FINDING_CONFIG).unwrap())
            }
            PartitioningStrategy::Custom(path) => {
                context.configure_from_file(CString::new(path.to_str().unwrap()).unwrap())
            }
        }
    }
}
