
pub type Example<T> = Vec<T>;
pub type Dataset<T> = Vec<Example<T>>;

pub trait Classifier {
    /// Type of a single training example.
    type ExampleType; 

    /// Type of the labels output by the classifier.
    type LabelType;

    /// Train the classifier on given labeled data.
    fn fit(&mut self, data: Vec<Self::ExampleType>, labels: Vec<Self::LabelType>);

    /// Predict the labels of a bunch of datapoints.
    fn predict(&self, data: &Vec<Self::ExampleType>) -> Option<Vec<Self::LabelType>>;

    /// Predict the label of one datapoint.
    fn predict_one(&self, x: &Self::ExampleType) -> Option<Self::LabelType>;
}
