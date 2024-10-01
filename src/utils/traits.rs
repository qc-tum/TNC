use std::fmt::Debug;
use std::{
    collections::HashMap,
    hash::{BuildHasher, Hash},
};

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
        self.try_insert(key, value).unwrap();
    }
}

#[cfg(test)]
mod tests {
    use rustc_hash::FxHashMap;

    use super::HashMapInsertNew;

    #[test]
    fn test_insert_new() {
        let mut hm = FxHashMap::default();
        hm.insert_new(1, 2);
        hm.insert_new(2, 4);
        assert_eq!(hm[&1], 2);
        assert_eq!(hm[&2], 4);
    }

    #[test]
    #[should_panic(
        expected = "called `Result::unwrap()` on an `Err` value: OccupiedError { key: 1, old_value: 2, new_value: 4, .. }"
    )]
    fn test_insert_new_panic() {
        let mut hm = FxHashMap::default();
        hm.insert_new(1, 2);
        // try to udpate value:
        hm.insert_new(1, 4);
    }
}
