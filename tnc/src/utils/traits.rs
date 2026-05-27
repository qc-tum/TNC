use std::collections::hash_map::Entry;
use std::fmt::Debug;
use std::{
    collections::HashMap,
    hash::{BuildHasher, Hash},
};

use ndarray::ArrayD;
use num_complex::Complex64;
use permutation::Permutation;
use rustc_hash::{FxBuildHasher, FxHashMap};

/// Trait for inserting a new key-value pair into a map.
pub trait HashMapInsertNew<K, V> {
    /// Inserts a new key-value pair into the map.
    ///
    /// # Panics
    /// Panics if the key is already present in the map.
    fn insert_new(&mut self, key: K, value: V);
}

impl<K, V, H> HashMapInsertNew<K, V> for HashMap<K, V, H>
where
    K: Eq + Hash + Debug,
    V: Debug,
    H: BuildHasher,
{
    #[inline]
    fn insert_new(&mut self, key: K, value: V) {
        match self.entry(key) {
            Entry::Occupied(occupied_entry) => panic!(
                "can not insert value {value:?}, because there is already an entry ({:?}, {:?})",
                occupied_entry.key(),
                occupied_entry.get()
            ),
            Entry::Vacant(vacant_entry) => {
                vacant_entry.insert(value);
            }
        }
    }
}

/// Trait for objects that can be created with a given capacity.
pub trait WithCapacity {
    fn with_capacity(capacity: usize) -> Self;
}

impl<K, V> WithCapacity for FxHashMap<K, V> {
    #[inline]
    fn with_capacity(capacity: usize) -> Self {
        FxHashMap::with_capacity_and_hasher(capacity, FxBuildHasher)
    }
}

/// Extension trait for Permutation to get a `Vec<usize>` usable for permuting axes
/// of an ndarray.
pub trait PermutationToVec {
    fn to_vec(&self) -> Vec<usize>;
}

impl PermutationToVec for Permutation {
    #[inline]
    fn to_vec(&self) -> Vec<usize> {
        (0..self.len()).map(|i| self.apply_inv_idx(i)).collect()
    }
}

/// Extension trait for ArrayD to compute the conjugate in-place.
pub trait ConjugateInPlace {
    fn conjugate(&mut self);
}

impl ConjugateInPlace for ArrayD<Complex64> {
    #[inline]
    fn conjugate(&mut self) {
        self.map_inplace(|x| *x = x.conj());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use rustc_hash::FxHashMap;

    #[test]
    fn test_insert_new() {
        let mut hm = FxHashMap::default();
        hm.insert_new(1, 2);
        hm.insert_new(2, 4);
        assert_eq!(hm[&1], 2);
        assert_eq!(hm[&2], 4);
    }

    #[test]
    #[should_panic(expected = "can not insert value 4, because there is already an entry (1, 2)")]
    fn test_insert_new_panic() {
        let mut hm = FxHashMap::default();
        hm.insert_new(1, 2);
        // try to udpate value:
        hm.insert_new(1, 4);
    }

    #[test]
    fn test_with_capacity() {
        let hm = FxHashMap::<char, usize>::with_capacity(10);
        assert!(hm.capacity() >= 10);
    }

    #[test]
    fn test_permutation_to_vec() {
        // Permutation:
        // 0 -> 2
        // 1 -> 0
        // 2 -> 1
        let perm = Permutation::oneline(vec![2, 0, 1]);
        // ndarray uses the convention that out[i] = in[perm[i]], e.g. if
        // in = [a, b, c], then out should be [b, c, a].
        // perm as vec should then be [1, 2, 0].

        assert_eq!(perm.to_vec(), vec![1, 2, 0]);
    }
}
