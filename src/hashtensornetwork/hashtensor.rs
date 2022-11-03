use linked_hash_set::LinkedHashSet;
use std::fmt;

#[derive(Eq, PartialEq, Debug, Clone)]
pub struct HashTensor {
    legs: LinkedHashSet<i32>,
}

impl HashTensor {
    pub fn new(legs: LinkedHashSet<i32>) -> Self {
        Self { legs }
    }

    pub fn get_legs(&self) -> &LinkedHashSet<i32> {
        &self.legs
    }

    pub fn iter(&self) -> linked_hash_set::Iter<'_, i32> {
        self.legs.iter()
    }
}

impl fmt::Display for HashTensor {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self.legs.iter())
    }
}
