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
}

#[cfg(test)]
mod tests {

    #[test]
    fn test_union_find_individual() {
        let uf = super::UnionFind::new(3);
        assert_eq!(uf.find(0), 0);
        assert_eq!(uf.find(1), 1);
        assert_eq!(uf.find(2), 2);
    }

    #[test]
    fn test_union_find_union() {
        let mut uf = super::UnionFind::new(3);
        uf.union(0, 1);
        assert_eq!(uf.find(0), uf.find(1));
        assert_ne!(uf.find(0), uf.find(2));
        assert_ne!(uf.find(1), uf.find(2));
    }

    #[test]
    fn test_union_find_union_skip() {
        let mut uf = super::UnionFind::new(3);
        uf.union(0, 2);
        assert_eq!(uf.find(0), uf.find(2));
        assert_ne!(uf.find(0), uf.find(1));
        assert_ne!(uf.find(2), uf.find(1));
    }

    #[test]
    fn test_union_find_union_multiple() {
        let mut uf = super::UnionFind::new(5);
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
}
