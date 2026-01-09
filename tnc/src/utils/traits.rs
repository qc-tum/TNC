use std::collections::hash_map::Entry;
use std::fmt::Debug;
use std::{
    collections::HashMap,
    hash::{BuildHasher, Hash},
};

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
}
