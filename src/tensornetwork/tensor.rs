use core::ops::{BitAnd, BitOr, BitXor, Sub};
use std::collections::HashMap;
use std::fmt;
use std::ops::{Index, IndexMut};

#[derive(Eq, PartialEq, Debug, Clone, Hash)]
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

    /// Returns number of dimensions of Tensor object
    ///
    /// # Examples
    /// ```
    /// use tensorcontraction::tensornetwork::tensor::Tensor;
    /// let vec = Vec::from([1,2,3]);
    /// let tensor = Tensor::new(vec.clone()) ;
    /// assert_eq!(tensor.dims(), 3);
    /// ```
    pub fn dims(&self) -> usize {
        self.legs.len()
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

    /// Returns true if Tensor contains leg_id
    ///
    /// # Arguments
    ///
    /// * `leg_id` - `usize` referencing specific leg
    /// ```
    fn contains(&self, leg_id: usize) -> bool {
        self.legs.contains(&leg_id)
    }

    /// Returns Tensor with legs in `self` that are not in `other`.
    ///
    /// # Arguments
    ///
    /// * `other` - Tensor with legs to remove
    ///
    /// # Examples
    /// ```
    /// use tensorcontraction::tensornetwork::tensor::Tensor;
    /// use std::collections::HashMap;
    /// let tensor1 = Tensor::new(vec![1,2,3]);
    /// let tensor2 = Tensor::new(vec![4,2,5]);
    /// let diff_tensor = &tensor1 - &tensor2;
    /// assert_eq!(diff_tensor, Tensor::new(vec![1,3]));
    /// ```
    pub fn difference(&self, other: &Tensor) -> Tensor {
        let mut new_legs = Vec::new();
        for &i in self.iter() {
            if !other.contains(i) {
                new_legs.push(i);
            }
        }
        Tensor::new(new_legs)
    }

    /// Returns Tensor with union of legs in both `self` and `other`.
    ///
    /// # Arguments
    ///
    /// * `other` - Tensor with legs to join
    ///
    /// # Examples
    /// ```
    /// use tensorcontraction::tensornetwork::tensor::Tensor;
    /// use std::collections::HashMap;
    /// let tensor1 = Tensor::new(vec![1,2,3]);
    /// let tensor2 = Tensor::new(vec![4,2,5]);
    /// let union_tensor = &tensor1 | &tensor2;
    /// assert_eq!(union_tensor, Tensor::new(vec![1,2,3,4,5]));
    /// ```
    pub fn union(&self, other: &Tensor) -> Tensor {
        let mut new_legs = self.legs.clone();
        for i in other.iter().cloned() {
            if !self.contains(i) {
                new_legs.push(i);
            }
        }
        Tensor::new(new_legs)
    }

    /// Returns Tensor with intersection of legs in `self` and `other`.
    ///
    /// # Arguments
    ///
    /// * `other` - Tensor with legs to intersect
    ///
    /// # Examples
    /// ```
    /// use tensorcontraction::tensornetwork::tensor::Tensor;
    /// use std::collections::HashMap;
    /// let tensor1 = Tensor::new(vec![1,2,3]);
    /// let tensor2 = Tensor::new(vec![4,2,5]);
    /// let intersection_tensor = &tensor1 & &tensor2;
    /// assert_eq!(intersection_tensor, Tensor::new(vec![2]));
    /// ```
    pub fn intersection(&self, other: &Tensor) -> Tensor {
        let mut new_legs = Vec::new();
        for i in self.iter().cloned() {
            if other.contains(i) {
                new_legs.push(i);
            }
        }
        Tensor::new(new_legs)
    }

    /// Returns Tensor with intersection of legs in `self` and `other`.
    ///
    /// # Arguments
    ///
    /// * `other` - Tensor with legs to intersect
    ///
    /// # Examples
    /// ```
    /// use tensorcontraction::tensornetwork::tensor::Tensor;
    /// use std::collections::HashMap;
    /// let tensor1 = Tensor::new(vec![1,2,3]);
    /// let tensor2 = Tensor::new(vec![4,2,5]);
    /// let sym_dif_tensor = &tensor1 ^ &tensor2;
    /// assert_eq!(sym_dif_tensor, Tensor::new(vec![1,3,4,5]));
    /// ```
    pub fn symmetric_difference(&self, other: &Tensor) -> Tensor {
        let mut new_legs = Vec::new();
        for i in self.iter().cloned() {
            if !other.contains(i) {
                new_legs.push(i);
            }
        }
        for i in other.iter().cloned() {
            if !self.contains(i) {
                new_legs.push(i);
            }
        }
        Tensor::new(new_legs)
    }
}

impl Default for Tensor {
    /// Constructs an empty Tensor object
    ///
    ///
    /// # Examples
    /// ```
    /// use tensorcontraction::tensornetwork::tensor::Tensor;
    /// let tensor = Tensor::default();
    /// ```
    fn default() -> Self {
        Self { legs: Vec::new() }
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

impl BitOr for &Tensor {
    type Output = Tensor;
    fn bitor(self, rhs: &Tensor) -> Tensor {
        self.union(rhs)
    }
}

impl BitAnd for &Tensor {
    type Output = Tensor;
    fn bitand(self, rhs: &Tensor) -> Tensor {
        self.intersection(rhs)
    }
}

impl BitXor for &Tensor {
    type Output = Tensor;
    fn bitxor(self, rhs: &Tensor) -> Tensor {
        self.symmetric_difference(rhs)
    }
}

impl Sub for &Tensor {
    type Output = Tensor;
    fn sub(self, rhs: &Tensor) -> Tensor {
        self.difference(rhs)
    }
}
