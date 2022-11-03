use std::fmt;
use crate::tensornetwork::Maximum;

#[derive(Eq, Ord, PartialEq, PartialOrd, Debug, Clone)]
pub struct Tensor {
    legs: Vec<i32>,
}


impl Maximum for Tensor{
    fn maximum(&self) -> i32 {
        let mut m = self.legs[0];
        for leg in self.legs.iter() {
            if *leg > m {
                m = *leg;
            }
        }
        m
    }
}


impl Tensor {
    pub fn new(legs: Vec<i32>) -> Self {
        Self { legs }
    }

    pub fn get_legs(&self) -> &Vec<i32> {
        &self.legs
    }
}

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self.legs)
    }
}
