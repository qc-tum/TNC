use std::{ops::DerefMut, mem::take};

use super::ast::{BinOp, Expr, UnOp};

fn simplify_copy(expr: Box<Expr>) -> Box<Expr> {
    match *expr {
        Expr::Unary(UnOp::Neg, inner) => {
            let inner = simplify_copy(inner);
            if let Expr::Unary(UnOp::Neg, inner2) = *inner {
                inner2
            } else {
                Box::new(Expr::Unary(UnOp::Neg, inner))
            }
        }
        Expr::Binary(op, lhs, rhs) => {
            let lhs = simplify_copy(lhs);
            let rhs = simplify_copy(rhs);
            match (&op, &*lhs, &*rhs) {
                (op, Expr::Float(a), Expr::Float(b)) => {
                    let res = match op {
                        BinOp::Add => a + b,
                        BinOp::Sub => a - b,
                        BinOp::Mul => a * b,
                        BinOp::Div => a / b,
                        BinOp::BitXor => panic!("BitXor is not defined on floats"),
                    };
                    Box::new(Expr::Float(res))
                }
                (op, Expr::Int(a), Expr::Int(b)) => {
                    let res = match op {
                        BinOp::Add => a + b,
                        BinOp::Sub => a - b,
                        BinOp::Mul => a * b,
                        BinOp::Div => a / b,
                        BinOp::BitXor => a | b,
                    };
                    Box::new(Expr::Int(res))
                }
                _ => Box::new(Expr::Binary(op, lhs, rhs)),
            }
        }
        _ => expr,
    }
}

fn simplify_inplace(expr: &mut Box<Expr>) {
    match expr.deref_mut() {
        Expr::Unary(UnOp::Neg, inner) => {
            simplify_inplace(inner);
            match inner.deref_mut() {
                Expr::Unary(UnOp::Neg, inner2) => {
                    *expr = take(inner2);
                },
                _ => (),
            }
        },
        _ => (),
    }
}

#[cfg(test)]
mod tests {
    use crate::qasm::{
        ast::{BinOp, Expr, UnOp},
        expression_folder::simplify_inplace,
    };

    use super::simplify_copy;

    #[test]
    fn copy() {
        let a = Expr::Unary(
            UnOp::Neg,
            Box::new(Expr::Unary(
                UnOp::Neg,
                Box::new(Expr::Unary(
                    UnOp::Neg,
                    Box::new(Expr::Binary(
                        BinOp::Add,
                        Box::new(Expr::Float(12.0)),
                        Box::new(Expr::Int(2)),
                    )),
                )),
            )),
        );
        println!("{a:?}");
        let b = simplify_copy(Box::new(a));
        println!("{b:?}");
    }

    #[test]
    fn inplace() {
        let mut a = Box::new(Expr::Unary(
            UnOp::Neg,
            Box::new(Expr::Unary(UnOp::Neg, Box::new(Expr::Int(2)))),
        ));
        println!("{a:?}");
        simplify_inplace(&mut a);
        println!("{a:?}");
    }
}
