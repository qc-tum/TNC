use std::cell::RefCell;

use itertools::Itertools;

/// Implements a union-find data structure.
/// This is also known as a disjoint-set data structure.
/// See <https://en.wikipedia.org/wiki/Disjoint-set_data_structure>.
#[derive(Debug)]
pub struct UnionFind {
    parent: RefCell<Vec<usize>>,
    rank: Vec<usize>,
}

impl UnionFind {
    /// Creates a new union-find data structure with `n` elements.
    #[must_use]
    pub fn new(n: usize) -> Self {
        let parent = (0..n).collect_vec();
        let rank = vec![0; n];
        Self {
            parent: RefCell::new(parent),
            rank,
        }
    }

    /// Finds the root element of the set containing `u`.
    /// This method uses *path splitting* as optimization.
    #[must_use]
    pub fn find(&self, u: usize) -> usize {
        let mut r = u;
        let mut parents = self.parent.borrow_mut();
        loop {
            if r == parents[r] {
                break;
            }

            parents[r] = parents[parents[r]];
            r = parents[r];
        }
        r
    }

    /// Unions the sets containing `x` and `y`.
    /// This method uses *union by rank* as optimization.
    pub fn union(&mut self, x: usize, y: usize) {
        // Find the root elements of the sets containing `x` and `y`
        let u = self.find(x);
        let v = self.find(y);

        if u == v {
            // Already in the same set
            return;
        }

        // Get low and high ranks
        let (low, high) = if self.rank[u] < self.rank[v] {
            (u, v)
        } else {
            (v, u)
        };

        // Merge the two sets
        self.parent.borrow_mut()[low] = high;
        if self.rank[low] == self.rank[high] {
            self.rank[high] += 1;
        }
    }

    /// Returns the number of elements in the union-find data structure.
    #[must_use]
    #[inline]
    pub fn len(&self) -> usize {
        self.rank.len()
    }

    /// Returns the number of disjoint sets in the union-find data structure.
    #[must_use]
    pub fn count_sets(&self) -> usize {
        (0..self.len()).map(|i| self.find(i)).unique().count()
    }
}

#[cfg(test)]
mod tests {
    use super::UnionFind;

    #[test]
    fn test_union_find_individual() {
        let uf = UnionFind::new(3);
        assert_eq!(uf.find(0), 0);
        assert_eq!(uf.find(1), 1);
        assert_eq!(uf.find(2), 2);
    }

    #[test]
    fn test_union_find_union() {
        let mut uf = UnionFind::new(3);
        uf.union(0, 1);
        assert_eq!(uf.find(0), uf.find(1));
        assert_ne!(uf.find(0), uf.find(2));
        assert_ne!(uf.find(1), uf.find(2));
    }

    #[test]
    fn test_union_find_union_skip() {
        let mut uf = UnionFind::new(3);
        uf.union(0, 2);
        assert_eq!(uf.find(0), uf.find(2));
        assert_ne!(uf.find(0), uf.find(1));
        assert_ne!(uf.find(2), uf.find(1));
    }

    #[test]
    fn test_union_find_union_multiple() {
        let mut uf = UnionFind::new(5);
        uf.union(1, 2);
        uf.union(3, 4);
        uf.union(2, 3);
        for a in 1..=4 {
            for b in 1..=4 {
                assert_eq!(uf.find(a), uf.find(b));
            }
            assert_ne!(uf.find(a), uf.find(0));
        }
    }

    #[test]
    fn test_union_find_len() {
        let uf = UnionFind::new(5);
        assert_eq!(uf.len(), 5);
    }

    #[test]
    fn test_union_find_len_doesnt_change() {
        let mut uf = UnionFind::new(5);
        uf.union(1, 2);
        assert_eq!(uf.len(), 5);
    }

    #[test]
    fn test_union_find_count_sets() {
        let mut uf = UnionFind::new(5);
        // (0), (1), (2), (3), (4)
        assert_eq!(uf.count_sets(), 5);
        uf.union(1, 2);
        // (0), (1, 2), (3), (4)
        assert_eq!(uf.count_sets(), 4);
        uf.union(3, 4);
        // (0), (1, 2), (3, 4)
        assert_eq!(uf.count_sets(), 3);
        uf.union(2, 3);
        // (0), (1, 2, 3, 4)
        assert_eq!(uf.count_sets(), 2);
        uf.union(0, 1);
        // (0, 1, 2, 3, 4)
        assert_eq!(uf.count_sets(), 1);
    }
}
