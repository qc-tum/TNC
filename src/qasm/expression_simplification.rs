use std::mem::take;
use std::ops::DerefMut;

use super::ast::{BinOp, Expr, FuncType, UnOp};

pub fn simplify(expr: &mut Expr) {
    match expr {
        Expr::Unary(UnOp::Neg, inner) => {
            simplify(inner);

            match inner.deref_mut() {
                // -(x) => (-x)
                Expr::Int(x) => {
                    *expr = Expr::Int(-*x);
                }
                // -(x) => (-x)
                Expr::Float(x) => {
                    *expr = Expr::Float(-*x);
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
                *expr = match op {
                    BinOp::Add => lhs.as_ref() + rhs.as_ref(),
                    BinOp::Sub => lhs.as_ref() - rhs.as_ref(),
                    BinOp::Mul => lhs.as_ref() * rhs.as_ref(),
                    BinOp::Div => lhs.as_ref() / rhs.as_ref(),
                    BinOp::BitXor => lhs.as_ref() ^ rhs.as_ref(),
                };
            }
            // TODO: Optimizations (like x + 0 = x)
        }

        Expr::Function(ftype, inner) => {
            simplify(inner);

            if inner.is_const() {
                // Get value as float (there are not functions that require an int)
                let val = match inner.deref_mut() {
                    Expr::Int(x) => (*x).into(),
                    Expr::Float(x) => *x,
                    _ => panic!("Expression is const, but neither float nor int"),
                };

                *expr = Expr::Float(match ftype {
                    FuncType::Sin => val.sin(),
                    FuncType::Cos => val.cos(),
                    FuncType::Tan => val.tan(),
                    FuncType::Exp => val.exp(),
                    FuncType::Ln => val.ln(),
                    FuncType::Sqrt => val.sqrt(),
                });
            }
        }

        _ => (),
    }
}

#[cfg(test)]
mod tests {
    use tux::assert_panic;

    use crate::qasm::{
        ast::{BinOp, Expr, FuncType, UnOp},
        expression_simplification::simplify,
    };

    #[test]
    fn long_expression() {
        // -(-4 * sin(2) / (2 - cos(-1.3)))
        let mut a = Expr::Unary(
            UnOp::Neg,
            Box::new(Expr::Binary(
                BinOp::Div,
                Box::new(Expr::Binary(
                    BinOp::Mul,
                    Box::new(Expr::Unary(UnOp::Neg, Box::new(Expr::Int(4)))),
                    Box::new(Expr::Function(FuncType::Sin, Box::new(Expr::Int(2)))),
                )),
                Box::new(Expr::Binary(
                    BinOp::Sub,
                    Box::new(Expr::Int(2)),
                    Box::new(Expr::Function(
                        FuncType::Cos,
                        Box::new(Expr::Unary(UnOp::Neg, Box::new(Expr::Float(1.3)))),
                    )),
                )),
            )),
        );
        simplify(&mut a);
        assert_eq!(
            a,
            Expr::Float(-(-4.0 * 2f64.sin() / (2.0 - (-1.3f64).cos())))
        );
    }

    #[test]
    fn neg_neg() {
        let mut a = Expr::Unary(
            UnOp::Neg,
            Box::new(Expr::Unary(UnOp::Neg, Box::new(Expr::Int(64)))),
        );
        simplify(&mut a);
        assert_eq!(a, Expr::Int(64));
    }

    #[test]
    fn neg_neg_neg_int() {
        let mut a = Expr::Unary(
            UnOp::Neg,
            Box::new(Expr::Unary(
                UnOp::Neg,
                Box::new(Expr::Unary(UnOp::Neg, Box::new(Expr::Int(2)))),
            )),
        );
        simplify(&mut a);
        assert_eq!(a, Expr::Int(-2));
    }

    #[test]
    fn neg_neg_neg_float() {
        let mut a = Expr::Unary(
            UnOp::Neg,
            Box::new(Expr::Unary(
                UnOp::Neg,
                Box::new(Expr::Unary(UnOp::Neg, Box::new(Expr::Float(3.14)))),
            )),
        );
        simplify(&mut a);
        assert_eq!(a, Expr::Float(-3.14));
    }

    #[test]
    fn add_int() {
        let mut a = Expr::Binary(BinOp::Add, Box::new(Expr::Int(2)), Box::new(Expr::Int(4)));
        simplify(&mut a);
        assert_eq!(a, Expr::Int(6));
    }

    #[test]
    fn add_float() {
        let mut a = Expr::Binary(
            BinOp::Add,
            Box::new(Expr::Binary(
                BinOp::Add,
                Box::new(Expr::Float(0.1)),
                Box::new(Expr::Float(0.1)),
            )),
            Box::new(Expr::Float(0.1)),
        );
        simplify(&mut a);
        assert_eq!(a, Expr::Float(0.1 + 0.1 + 0.1));
    }

    #[test]
    fn add_mixed() {
        let mut a = Expr::Binary(
            BinOp::Add,
            Box::new(Expr::Float(0.5)),
            Box::new(Expr::Int(2)),
        );
        simplify(&mut a);
        assert_eq!(a, Expr::Float(0.5 + 2.0));
    }

    #[test]
    fn sub_int() {
        let mut a = Expr::Binary(BinOp::Sub, Box::new(Expr::Int(2)), Box::new(Expr::Int(4)));
        simplify(&mut a);
        assert_eq!(a, Expr::Int(-2));
    }

    #[test]
    fn sub_float() {
        let mut a = Expr::Binary(
            BinOp::Sub,
            Box::new(Expr::Binary(
                BinOp::Sub,
                Box::new(Expr::Float(0.1)),
                Box::new(Expr::Float(0.1)),
            )),
            Box::new(Expr::Float(0.1)),
        );
        simplify(&mut a);
        assert_eq!(a, Expr::Float(0.1 - 0.1 - 0.1));
    }

    #[test]
    fn sub_mixed() {
        let mut a = Expr::Binary(
            BinOp::Sub,
            Box::new(Expr::Float(0.5)),
            Box::new(Expr::Int(2)),
        );
        simplify(&mut a);
        assert_eq!(a, Expr::Float(0.5 - 2.0));
    }

    #[test]
    fn multiply_ints() {
        let mut a = Expr::Binary(BinOp::Mul, Box::new(Expr::Int(2)), Box::new(Expr::Int(3)));
        simplify(&mut a);
        assert_eq!(a, Expr::Int(2 * 3));
    }

    #[test]
    fn multiply_floats() {
        let mut a = Expr::Binary(
            BinOp::Mul,
            Box::new(Expr::Float(3.6)),
            Box::new(Expr::Float(-2.4)),
        );
        simplify(&mut a);
        assert_eq!(a, Expr::Float(3.6 * -2.4));
    }

    #[test]
    fn multiply_mixed() {
        let mut a = Expr::Binary(
            BinOp::Mul,
            Box::new(Expr::Int(3)),
            Box::new(Expr::Float(0.5)),
        );
        simplify(&mut a);
        assert_eq!(a, Expr::Float(3f64 * 0.5));
    }

    #[test]
    fn divide_ints() {
        let mut a = Expr::Binary(BinOp::Div, Box::new(Expr::Int(30)), Box::new(Expr::Int(4)));
        simplify(&mut a);
        assert_eq!(a, Expr::Int(30 / 4));
    }

    #[test]
    fn divide_floats() {
        let mut a = Expr::Binary(
            BinOp::Div,
            Box::new(Expr::Float(3.6)),
            Box::new(Expr::Float(-2.4)),
        );
        simplify(&mut a);
        assert_eq!(a, Expr::Float(3.6 / -2.4));
    }

    #[test]
    fn divide_mixed() {
        let mut a = Expr::Binary(
            BinOp::Div,
            Box::new(Expr::Int(3)),
            Box::new(Expr::Float(0.5)),
        );
        simplify(&mut a);
        assert_eq!(a, Expr::Float(3f64 / 0.5));
    }

    #[test]
    fn xor_float_fail() {
        let mut a = Expr::Binary(
            BinOp::BitXor,
            Box::new(Expr::Float(1.0)),
            Box::new(Expr::Int(2)),
        );
        assert_panic!("Cannot apply bitxor on non-int types" in simplify(&mut a));
    }

    #[test]
    fn xor_int() {
        let mut a = Expr::Binary(
            BinOp::BitXor,
            Box::new(Expr::Int(5)),
            Box::new(Expr::Int(2)),
        );
        simplify(&mut a);
        assert_eq!(a, Expr::Int(7));
    }

    #[test]
    fn sqrt_int() {
        let mut a = Expr::Function(FuncType::Sqrt, Box::new(Expr::Int(4)));
        simplify(&mut a);
        assert_eq!(a, Expr::Float(4f64.sqrt()));
    }

    #[test]
    fn cos_float() {
        let mut a = Expr::Function(FuncType::Cos, Box::new(Expr::Float(2.3)));
        simplify(&mut a);
        assert_eq!(a, Expr::Float(2.3f64.cos()));
    }
}
