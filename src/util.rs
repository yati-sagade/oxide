use std::collections::HashMap;
use std::collections::hash_map::Iter;
use std::hash::Hash;
use std::num;

/// Compute the Euclidean distance between two vectors.
pub fn euclidean_distance(v1: &[f64], v2: &[f64]) -> f64 {
    squared_distance(v1, v2).sqrt()
}

/// Compute the squared norm of the vector difference of v1 and v2.
pub fn squared_distance(v1: &[f64], v2: &[f64]) -> f64 {
    let delta: Vec<_> = v1.iter().zip(v2.iter()).map(|(a, b)| { a - b }).collect();
    dot_product(&delta, &delta)
}

/// Compute the dot product of two vectors.
pub fn dot_product(v1: &[f64], v2: &[f64]) -> f64 {
    v1.iter().zip(v2.iter()).fold(0f64, |acc, (a, b)| {
        acc + a * b
    })
}

/// An item counter, similar to Python's collections.Counter.
pub struct Counter<T> {
    map: HashMap<T, u64>
}

impl<T: Hash + Eq> Counter<T> {
    /// Construct an empty Counter.
    pub fn new() -> Counter<T> {
        Counter::<T> { map: HashMap::new() }
    }

    /// Construct a Counter from an iterator.
    pub fn with_iterator<U>(it: U) -> Counter<T> where U: Iterator<Item=T> {
        let mut ctr = Counter::new();
        for item in it {
            ctr.insert(item);
        }
        ctr
    }

    /// Insert an item in the counter, increasing its count by one.
    pub fn insert(&mut self, item: T) {
        let new_val = match self.map.get(&item) {
            Some(val) => val + 1,
            None      => 1
        };
        self.map.insert(item, new_val);
    }

    /// Get the count of an item.
    pub fn get(&self, item: &T) -> Option<u64> {
        match self.map.get(item) {
            Some(val) => Some(*val),
            None      => None,
        }
    }

    /// Get the most frequent item and its frequency.
    pub fn most_frequent(&self) -> Option<(&T, u64)> {
        if self.map.is_empty() {
            return None;
        }
        let mut rval = None;
        let mut rfreq: u64 = 0;

        for (item, freq) in self.iter() {
            if *freq > rfreq {
                rfreq = *freq;
                rval = Some(item);
            }
        }
        Some((rval.unwrap(), rfreq))
    }

    /// Get an iterator over the counter.
    pub fn iter(&self) -> Iter<T, u64> {
        self.map.iter()
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_counter() {
        let mut ctr: Counter<bool> = Counter::new();
        ctr.insert(true);
        ctr.insert(true);
        ctr.insert(false);

        assert_eq!(ctr.get(&true).unwrap(), 2u64);
        assert_eq!(ctr.get(&false).unwrap(), 1u64);

        assert_eq!(ctr.most_frequent().unwrap(), (&true, 2u64));
    }

    #[test]
    fn dot_works() {
        let x = vec![1f64, 2.0, 3.0];
        let y = vec![2f64, 5.0, -1.0];
        let dot = dot_product(&x, &y);
        assert_eq!(dot, 9f64);
    }
}
