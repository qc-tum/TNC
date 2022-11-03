use std::fmt;

#[derive(Eq, Ord, PartialEq, PartialOrd, Debug, Clone)]
pub struct Tensor {
    legs: Vec<i32>,
}



impl Tensor {
    pub fn new(legs: Vec<i32>) -> Self {
        Self { legs }
    }

    pub fn get_legs(&self) -> &Vec<i32> {
        &self.legs
    }

    pub fn iter(&self) -> std::slice::Iter<'_,i32>{
        self.legs.iter()
    }
}

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self.legs)
    }
}
