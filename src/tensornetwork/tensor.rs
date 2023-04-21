use std::collections::HashMap;
use std::fmt;
use std::ops::{Index, IndexMut};

#[derive(Eq, PartialEq, Debug, Clone)]
/// Abstract representation of a tensor. Stores a Vector of edge ids, used to indicate
/// contractions between Tensors. Edge dimensions are stored in a separate HashMap object.
/// See [TensorNetwork].
pub struct Tensor {
    /// Stores edge ids in a Vector.
    legs: Vec<usize>,
}

impl Tensor {
    /// Constructs a Tensor object
    ///
    /// # Arguments
    ///
    /// * `legs` - A vector of usize containing edge ids.
    ///
    /// # Examples
    /// ```
    /// use tensorcontraction::tensornetwork::tensor::Tensor;
    /// let vec = Vec::from([1,2,3]);
    /// let tensor = Tensor::new(vec);
    /// ```
    pub fn new(legs: Vec<usize>) -> Self {
        Self { legs }
    }
    /// Returns edge ids of Tensor object
    ///
    /// # Examples
    /// ```
    /// use tensorcontraction::tensornetwork::tensor::Tensor;
    /// let vec = Vec::from([1,2,3]);
    /// let tensor = Tensor::new(vec.clone()) ;
    /// assert_eq!(*tensor.get_legs(), vec);
    /// ```
    pub fn get_legs(&self) -> &Vec<usize> {
        &self.legs
    }

    /// Returns Iter of Tensor object legs
    ///
    /// # Examples
    /// ```
    /// use tensorcontraction::tensornetwork::tensor::Tensor;
    /// let vec = Vec::from([1,2,3]);
    /// let tensor = Tensor::new(vec.clone()) ;
    /// assert_eq!(tensor.iter().eq(vec.iter()), true);
    /// ```
    pub fn iter(&self) -> std::slice::Iter<'_, usize> {
        self.legs.iter()
    }

    /// Returns product of leg sizes based on input Hashmap. Returns the number of elements in a tensor
    ///
    /// # Arguments
    ///
    /// * `bond_dim` - Reference to hashmap mapping edge ID to bond dimension size
    ///
    /// # Examples
    /// ```
    /// use tensorcontraction::tensornetwork::tensor::Tensor;
    /// use std::collections::HashMap;
    /// let vec = Vec::from([1,2,3]);
    /// let tensor = Tensor::new(vec.clone()) ;
    /// let mut hm = HashMap::new();
    /// hm.insert(1, 5);
    /// hm.insert(2, 15);
    /// hm.insert(3, 8);
    /// assert_eq!(tensor.size(&hm), 600);
    /// ```
    pub fn size(&self, bond_dim: &HashMap<usize, u64>) -> u64 {
        self.legs.iter().map(|e| bond_dim[e]).product()
    }
}

/// Implementation of printing for Tensor. Simply prints the legs as a vector
impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self.legs)
    }
}

/// Implementation of indexing for Tensor.
impl Index<usize> for Tensor {
    type Output = usize;

    fn index(&self, index: usize) -> &Self::Output {
        &self.legs[index]
    }
}

/// Implementation of indexing of mutable Tensor object.
impl IndexMut<usize> for Tensor {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.legs[index]
    }
}
