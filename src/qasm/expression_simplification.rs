use std::mem::take;

use super::ast::{BinOp, Expr, FuncType, UnOp};

/// Folds (i.e. computes) constant subexpressions in the expression. For instance,
/// `a * (5 + sin(0))` becomes `a * 5`.
/// Does not perform optimizations (e.g. `a + 0` is left unchanged).
pub fn fold_expr(expr: &mut Expr) {
    match expr {
        Expr::Unary(UnOp::Neg, inner) => {
            fold_expr(inner);

            match &mut **inner {
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
            fold_expr(lhs);
            fold_expr(rhs);

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
        }

        Expr::Function(ftype, inner) => {
            fold_expr(inner);

            if inner.is_const() {
                // Get value as float (there are no functions that require an int)
                let val: f64 = (&**inner).try_into().unwrap();

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
    use crate::qasm::{
        ast::{BinOp, Expr, FuncType, UnOp},
        expression_simplification::fold_expr,
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
        fold_expr(&mut a);
        assert_eq!(
            a,
            Expr::Float(-(-4.0 * 2f64.sin() / (2.0 - (-1.3f64).cos())))
        );
    }

    #[test]
    fn neg_neg() {
        // -(-64) = 64
        let mut a = Expr::Unary(
            UnOp::Neg,
            Box::new(Expr::Unary(UnOp::Neg, Box::new(Expr::Int(64)))),
        );
        fold_expr(&mut a);
        assert_eq!(a, Expr::Int(64));
    }

    #[test]
    fn neg_neg_neg_int() {
        // -(-(-2)) = -2
        let mut a = Expr::Unary(
            UnOp::Neg,
            Box::new(Expr::Unary(
                UnOp::Neg,
                Box::new(Expr::Unary(UnOp::Neg, Box::new(Expr::Int(2)))),
            )),
        );
        fold_expr(&mut a);
        assert_eq!(a, Expr::Int(-2));
    }

    #[test]
    fn neg_neg_neg_float() {
        // -(-(-0.1)) = -0.1
        let mut a = Expr::Unary(
            UnOp::Neg,
            Box::new(Expr::Unary(
                UnOp::Neg,
                Box::new(Expr::Unary(UnOp::Neg, Box::new(Expr::Float(0.1)))),
            )),
        );
        fold_expr(&mut a);
        assert_eq!(a, Expr::Float(-0.1));
    }

    #[test]
    fn add_int() {
        // 2 + 4
        let mut a = Expr::Binary(BinOp::Add, Box::new(Expr::Int(2)), Box::new(Expr::Int(4)));
        fold_expr(&mut a);
        assert_eq!(a, Expr::Int(6));
    }

    #[test]
    fn add_float() {
        // 0.1 + (0.1 + 0.1)
        let mut a = Expr::Binary(
            BinOp::Add,
            Box::new(Expr::Binary(
                BinOp::Add,
                Box::new(Expr::Float(0.1)),
                Box::new(Expr::Float(0.1)),
            )),
            Box::new(Expr::Float(0.1)),
        );
        fold_expr(&mut a);
        assert_eq!(a, Expr::Float(0.1 + 0.1 + 0.1));
    }

    #[test]
    fn add_mixed() {
        // 0.5 + 2
        let mut a = Expr::Binary(
            BinOp::Add,
            Box::new(Expr::Float(0.5)),
            Box::new(Expr::Int(2)),
        );
        fold_expr(&mut a);
        assert_eq!(a, Expr::Float(0.5 + 2.0));
    }

    #[test]
    fn sub_int() {
        // 2 - 4
        let mut a = Expr::Binary(BinOp::Sub, Box::new(Expr::Int(2)), Box::new(Expr::Int(4)));
        fold_expr(&mut a);
        assert_eq!(a, Expr::Int(-2));
    }

    #[test]
    fn sub_float() {
        // (0.1 - 0.1) - 0.1
        let mut a = Expr::Binary(
            BinOp::Sub,
            Box::new(Expr::Binary(
                BinOp::Sub,
                Box::new(Expr::Float(0.1)),
                Box::new(Expr::Float(0.1)),
            )),
            Box::new(Expr::Float(0.1)),
        );
        fold_expr(&mut a);
        assert_eq!(a, Expr::Float(0.1 - 0.1 - 0.1));
    }

    #[test]
    fn sub_mixed() {
        // 0.5 - 2
        let mut a = Expr::Binary(
            BinOp::Sub,
            Box::new(Expr::Float(0.5)),
            Box::new(Expr::Int(2)),
        );
        fold_expr(&mut a);
        assert_eq!(a, Expr::Float(0.5 - 2.0));
    }

    #[test]
    fn multiply_ints() {
        // 2 * 3
        let mut a = Expr::Binary(BinOp::Mul, Box::new(Expr::Int(2)), Box::new(Expr::Int(3)));
        fold_expr(&mut a);
        assert_eq!(a, Expr::Int(2 * 3));
    }

    #[test]
    fn multiply_floats() {
        // 3.6 * -2.4
        let mut a = Expr::Binary(
            BinOp::Mul,
            Box::new(Expr::Float(3.6)),
            Box::new(Expr::Float(-2.4)),
        );
        fold_expr(&mut a);
        assert_eq!(a, Expr::Float(3.6 * -2.4));
    }

    #[test]
    fn multiply_mixed() {
        // 3 * 0.5
        let mut a = Expr::Binary(
            BinOp::Mul,
            Box::new(Expr::Int(3)),
            Box::new(Expr::Float(0.5)),
        );
        fold_expr(&mut a);
        assert_eq!(a, Expr::Float(3f64 * 0.5));
    }

    #[test]
    fn divide_ints() {
        // 30 / 4
        let mut a = Expr::Binary(BinOp::Div, Box::new(Expr::Int(30)), Box::new(Expr::Int(4)));
        fold_expr(&mut a);
        assert_eq!(a, Expr::Int(30 / 4));
    }

    #[test]
    fn divide_floats() {
        // 3.6 / -2.4
        let mut a = Expr::Binary(
            BinOp::Div,
            Box::new(Expr::Float(3.6)),
            Box::new(Expr::Float(-2.4)),
        );
        fold_expr(&mut a);
        assert_eq!(a, Expr::Float(3.6 / -2.4));
    }

    #[test]
    fn divide_mixed() {
        // 3 / 0.5
        let mut a = Expr::Binary(
            BinOp::Div,
            Box::new(Expr::Int(3)),
            Box::new(Expr::Float(0.5)),
        );
        fold_expr(&mut a);
        assert_eq!(a, Expr::Float(3f64 / 0.5));
    }

    #[test]
    #[should_panic(expected = "Cannot apply bitxor on non-int types")]
    fn xor_float_fail() {
        // 1.0 ^ 2 fails
        let mut a = Expr::Binary(
            BinOp::BitXor,
            Box::new(Expr::Float(1.0)),
            Box::new(Expr::Int(2)),
        );
        fold_expr(&mut a);
    }

    #[test]
    fn xor_int() {
        // 5 ^ 2
        let mut a = Expr::Binary(
            BinOp::BitXor,
            Box::new(Expr::Int(5)),
            Box::new(Expr::Int(2)),
        );
        fold_expr(&mut a);
        assert_eq!(a, Expr::Int(7));
    }

    #[test]
    fn sqrt_int() {
        // sqrt(4)
        let mut a = Expr::Function(FuncType::Sqrt, Box::new(Expr::Int(4)));
        fold_expr(&mut a);
        assert_eq!(a, Expr::Float(4f64.sqrt()));
    }

    #[test]
    fn cos_float() {
        // cos(2.3)
        let mut a = Expr::Function(FuncType::Cos, Box::new(Expr::Float(2.3)));
        fold_expr(&mut a);
        assert_eq!(a, Expr::Float(2.3f64.cos()));
    }
}
