pub type EdgeIndex = usize;
pub type TensorIndex = usize;

#[derive(Debug, Clone, Eq, PartialEq)]
pub enum Vertex {
    Open,
    Closed(TensorIndex),
}
