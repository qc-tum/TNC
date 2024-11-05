use std::fmt;
use std::rc::Weak;

use std::cell::RefCell;

use std::rc::Rc;

pub(crate) type NodeRef = Rc<RefCell<Node>>;
pub(crate) type WeakNodeRef = Weak<RefCell<Node>>;

/// Node in [`ContractionTree`], represents a contraction of [`Tensor`] with position
/// `left_child` and position `right_child` to obtain [`Tensor`] at position
/// `parent`.
#[derive(Debug, Clone)]
pub struct Node {
    id: usize,
    left_child: WeakNodeRef,
    right_child: WeakNodeRef,
    parent: WeakNodeRef,
    tensor_index: Option<Vec<usize>>,
}

impl Node {
    pub(crate) fn new(
        id: usize,
        left_child: WeakNodeRef,
        right_child: WeakNodeRef,
        parent: WeakNodeRef,
        tensor_index: Option<Vec<usize>>,
    ) -> Self {
        Self {
            id,
            left_child,
            right_child,
            parent,
            tensor_index,
        }
    }

    pub fn left_child_id(&self) -> Option<usize> {
        self.left_child.upgrade().map(|node| node.borrow().id)
    }

    pub fn right_child_id(&self) -> Option<usize> {
        self.right_child.upgrade().map(|node| node.borrow().id)
    }

    pub fn left_child(&self) -> Option<NodeRef> {
        self.left_child.upgrade()
    }

    pub fn right_child(&self) -> Option<NodeRef> {
        self.right_child.upgrade()
    }

    pub fn tensor_index(&self) -> &Option<Vec<usize>> {
        &self.tensor_index
    }

    pub const fn id(&self) -> usize {
        self.id
    }

    pub fn parent_id(&self) -> Option<usize> {
        self.parent.upgrade().map(|node| node.borrow().id)
    }

    pub(crate) fn is_leaf(&self) -> bool {
        self.left_child.upgrade().is_none() && self.right_child.upgrade().is_none()
    }

    pub(crate) fn remove_parent(&mut self) {
        self.parent = Default::default();
    }

    pub(crate) fn set_parent(&mut self, parent: WeakNodeRef) {
        assert!(
            self.parent.upgrade().is_none(),
            "Parent is already allocated"
        );
        self.parent = parent;
    }

    pub(crate) fn add_child(&mut self, child: WeakNodeRef) {
        assert!(child.upgrade().is_some(), "Child is already deallocated");
        if self.left_child.upgrade().is_none() {
            self.left_child = child;
        } else if self.right_child.upgrade().is_none() {
            self.right_child = child;
        } else {
            panic!("Parent already has two children");
        }
    }

    pub(crate) fn remove_child(&mut self, id: usize) {
        if let Some(left_id) = self.left_child_id() {
            if left_id == id {
                self.left_child = Default::default();
                return;
            }
        }
        if let Some(right_id) = self.right_child_id() {
            if right_id == id {
                self.right_child = Default::default();
                return;
            }
        }
        panic!("No child with id {id} found.");
    }
}

pub(crate) fn child_node(id: usize, tensor_index: Vec<usize>) -> Rc<RefCell<Node>> {
    Rc::new(RefCell::new(Node::new(
        id,
        Weak::new(),
        Weak::new(),
        Weak::new(),
        Some(tensor_index),
    )))
}

pub(crate) fn parent_node(
    id: usize,
    left_child: &Rc<RefCell<Node>>,
    right_child: &Rc<RefCell<Node>>,
) -> Rc<RefCell<Node>> {
    let parent = Rc::new(RefCell::new(Node::new(
        id,
        Rc::downgrade(left_child),
        Rc::downgrade(right_child),
        Weak::new(),
        None,
    )));
    left_child.borrow_mut().set_parent(Rc::downgrade(&parent));
    right_child.borrow_mut().set_parent(Rc::downgrade(&parent));
    parent
}

impl fmt::Display for Node {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Node {{ id: {}, left_child: {:?}, right_child: {:?}, parent: {:?}, tensor_index: {:?} }}",
            self.id,
            self.left_child_id(),
            self.right_child_id(),
            self.parent_id(),
            self.tensor_index
        )
    }
}

#[cfg(test)]
mod tests {

    use crate::contractionpath::contraction_tree::node::{child_node, parent_node};

    #[test]
    fn test_node_format() {
        let node3 = child_node(0, vec![0]);
        let node2 = child_node(1, vec![1]);
        let node5 = parent_node(5, &node3, &node2);
        let node_borrow = node5.borrow();

        assert_eq!(
            "Node { id: 5, left_child: Some(0), right_child: Some(1), parent: None, tensor_index: None }",
            node_borrow.to_string()
        )
    }
}
