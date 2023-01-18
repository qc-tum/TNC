use std::mem::take;
use std::ops::DerefMut;

use super::ast::{BinOp, Expr, UnOp};

fn simplify(expr: &mut Box<Expr>) {
    match expr.deref_mut() {
        Expr::Unary(UnOp::Neg, inner) => {
            simplify(inner);

            match inner.deref_mut() {
                // -(x) => (-x)
                Expr::Int(x) => {
                    *expr = Box::new(Expr::Int(-*x));
                }
                // -(x) => (-x)
                Expr::Float(x) => {
                    *expr = Box::new(Expr::Float(-*x));
                }
                // --x => x
                Expr::Unary(UnOp::Neg, inner2) => {
                    *expr = take(inner2);
                }
                _ => (),
            }
        }

        Expr::Binary(op, lhs, rhs) => {
            simplify(lhs);
            simplify(rhs);

            // Constant evaluation
            if lhs.is_const() && rhs.is_const() {
                *expr = Box::new(match op {
                    BinOp::Add => lhs.as_ref() + rhs.as_ref(),
                    BinOp::Sub => lhs.as_ref() - rhs.as_ref(),
                    BinOp::Mul => todo!(),
                    BinOp::Div => todo!(),
                    BinOp::BitXor => lhs.as_ref() ^ rhs.as_ref(),
                });
            }
        }

        _ => (),
    }
}

#[cfg(test)]
mod tests {
    use crate::qasm::{
        ast::{BinOp, Expr, UnOp},
        expression_folder::simplify,
    };

    #[test]
    fn neg_neg() {
        let mut a = Box::new(Expr::Unary(
            UnOp::Neg,
            Box::new(Expr::Unary(UnOp::Neg, Box::new(Expr::Int(64)))),
        ));
        simplify(&mut a);
        assert_eq!(*a, Expr::Int(64));
    }

    #[test]
    fn neg_neg_neg_int() {
        let mut a = Box::new(Expr::Unary(
            UnOp::Neg,
            Box::new(Expr::Unary(
                UnOp::Neg,
                Box::new(Expr::Unary(UnOp::Neg, Box::new(Expr::Int(2)))),
            )),
        ));
        simplify(&mut a);
        assert_eq!(*a, Expr::Int(-2));
    }

    #[test]
    fn neg_neg_neg_float() {
        let mut a = Box::new(Expr::Unary(
            UnOp::Neg,
            Box::new(Expr::Unary(
                UnOp::Neg,
                Box::new(Expr::Unary(UnOp::Neg, Box::new(Expr::Float(3.14)))),
            )),
        ));
        simplify(&mut a);
        assert_eq!(*a, Expr::Float(-3.14));
    }

    #[test]
    fn add_int() {
        let mut a = Box::new(Expr::Binary(
            BinOp::Add,
            Box::new(Expr::Int(2)),
            Box::new(Expr::Int(4)),
        ));
        simplify(&mut a);
        assert_eq!(*a, Expr::Int(6));
    }

    #[test]
    fn add_float() {
        let mut a = Box::new(Expr::Binary(
            BinOp::Add,
            Box::new(Expr::Binary(
                BinOp::Add,
                Box::new(Expr::Float(0.1)),
                Box::new(Expr::Float(0.1)),
            )),
            Box::new(Expr::Float(0.1)),
        ));
        simplify(&mut a);
        assert_eq!(*a, Expr::Float(0.1 + 0.1 + 0.1));
    }

    #[test]
    fn add_mixed() {
        let mut a = Box::new(Expr::Binary(
            BinOp::Add,
            Box::new(Expr::Float(0.5)),
            Box::new(Expr::Int(2)),
        ));
        simplify(&mut a);
        assert_eq!(*a, Expr::Float(0.5 + 2.0));
    }

    #[test]
    fn sub_int() {
        let mut a = Box::new(Expr::Binary(
            BinOp::Sub,
            Box::new(Expr::Int(2)),
            Box::new(Expr::Int(4)),
        ));
        simplify(&mut a);
        assert_eq!(*a, Expr::Int(-2));
    }

    #[test]
    fn sub_float() {
        let mut a = Box::new(Expr::Binary(
            BinOp::Sub,
            Box::new(Expr::Binary(
                BinOp::Sub,
                Box::new(Expr::Float(0.1)),
                Box::new(Expr::Float(0.1)),
            )),
            Box::new(Expr::Float(0.1)),
        ));
        simplify(&mut a);
        assert_eq!(*a, Expr::Float(0.1 - 0.1 - 0.1));
    }

    #[test]
    fn sub_mixed() {
        let mut a = Box::new(Expr::Binary(
            BinOp::Sub,
            Box::new(Expr::Float(0.5)),
            Box::new(Expr::Int(2)),
        ));
        simplify(&mut a);
        assert_eq!(*a, Expr::Float(0.5 - 2.0));
    }

    #[test]
    #[should_panic(expected = "Cannot apply bitxor on non-int types")]
    fn xor_float_fail() {
        let mut a = Box::new(Expr::Binary(
            BinOp::BitXor,
            Box::new(Expr::Float(1.0)),
            Box::new(Expr::Int(2)),
        ));
        simplify(&mut a);
    }

    #[test]
    fn xor_int() {
        let mut a = Box::new(Expr::Binary(
            BinOp::BitXor,
            Box::new(Expr::Int(5)),
            Box::new(Expr::Int(2)),
        ));
        simplify(&mut a);
        assert_eq!(*a, Expr::Int(7));
    }
}
