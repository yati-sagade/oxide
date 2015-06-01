use super::util::{Counter,squared_distance};
use std::hash::Hash;

/// A K-Nearest Neighbours classifier.
pub struct KNNClassifier<T> {
    k: usize,
    data: Option<Vec<Vec<f64>>>,
    labels: Option<Vec<T>>,
}

impl<T: Hash + Eq + Clone> KNNClassifier<T> {
    /// Construct a new KNNClassifier.
    pub fn new(k: usize) -> KNNClassifier<T> {
        KNNClassifier::<T>{ k: k, data: None, labels: None }
    }
    
    /// Train the classifier with examples and their labels. A KNN classifier
    /// doesn't actually do anything in the training phase, which is why it has
    /// been called a "lazy learner".
    pub fn fit(&mut self, data: Vec<Vec<f64>>, labels: Vec<T>) {
        self.data = Some(data);
        self.labels = Some(labels);
    }
    
    /// Predict the labels of datapoints. Return None if `predict()` is
    /// called before `fit()`.
    pub fn predict(&self, data: &Vec<Vec<f64>>) -> Option<Vec<T>> {
        if self.data.is_none() {
            return None;
        }
        let mut ret = Vec::with_capacity(data.len());
        for x_test in data {
            ret.push(self.predict_one(x_test).unwrap());
        }
        Some(ret)
    }
    
    /// Predict the label for one datapoint. Return None if `predict_one()`
    /// is called before `fit()`.
    pub fn predict_one(&self, x: &Vec<f64>) -> Option<T> {
        match self.data {
            Some(ref data) => {
                // Store the indices of the k nearest neighours so far.
                let mut best_neigh = Vec::with_capacity(self.k);
                let mut best_dists = Vec::with_capacity(self.k);
                for (i, x_train) in data.iter().enumerate() {
                    let dist = squared_distance(x, x_train);
                    if best_neigh.len() < self.k {
                        best_neigh.push(i);
                        best_dists.push(dist);
                    } else {
                        for j in 0..self.k {
                            // TODO: Use BTreeSet so that we can break out
                            // earlier here.
                            if dist < best_dists[j] {
                                best_dists[j] = dist;
                                best_neigh[j] = i;
                                break;
                            }
                        }
                    }
                }
                let ctr = match self.labels {
                    Some(ref labels) => Counter::with_iterator(best_neigh.iter().map(|&idx| {
                        labels[idx].clone()
                    })),
                    None             => panic!("Empty labels after training"),
                };
                let (ret, _): (&T, u64) = ctr.most_frequent().unwrap();
                Some((*ret).clone())
            },
            None => None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_creation() {
        let clf: KNNClassifier<String> = KNNClassifier::new(3);
        assert_eq!(clf.data, None);
        assert_eq!(clf.labels, None);
    }

    #[test]
    fn test_fit() {
        let mut clf = KNNClassifier::new(3);

        let train: Vec<Vec<f64>> = vec![
            vec![0.0, 1.0, 2.0, 2.0, 3.0],
            vec![5.0, 4.0, 3.0, 4.0, 5.0],
        ];

        let labels: Vec<String> = vec![
            "good".to_string(),
            "bad".to_string(),
        ];

        clf.fit(train.clone(), labels.clone());

        assert_eq!(clf.data, Some(train));
        assert_eq!(clf.labels, Some(labels));
    }

    #[test]
    fn test_predict() {
        let mut clf = KNNClassifier::new(1);

        let train: Vec<Vec<f64>> = vec![
            vec![0.0, 1.0, 2.0, 2.0, 3.0],
            vec![5.0, 4.0, 3.0, 4.0, 5.0],
            vec![0.0, 0.0, 0.0, 0.0, 0.0],
        ];

        let labels: Vec<String> = vec![
            "good".to_string(),
            "bad".to_string(),
            "good".to_string(),
        ];

        clf.fit(train.clone(), labels.clone());

        let test = vec![
            vec![1.0, 1.0, 1.0, 1.0, 1.0],
        ];

        let pred = clf.predict(&test).unwrap();
        
        assert_eq!(pred[0], "good".to_string());
    }
}
