use std::mem::take;

use crate::io::qasm::ast::{BinOp, Expr, FuncType, UnOp};

/// Folds (i.e. computes) constant subexpressions in the expression. For instance,
/// `a * (5 + sin(0))` becomes `a * 5`.
/// Does not perform optimizations (e.g. `a + 0` is left unchanged).
pub fn fold_expr(expr: &mut Expr) {
    match expr {
        Expr::Unary(UnOp::Neg, inner) => {
            fold_expr(inner);

            match &mut **inner {
                // -(x) => (-x)
                Expr::Const(x) => {
                    *expr = Expr::Const(-*x);
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
                    BinOp::Power => lhs.as_ref() ^ rhs.as_ref(),
                };
            }
        }

        Expr::Function(ftype, inner) => {
            fold_expr(inner);

            if inner.is_const() {
                // Get value as float (there are no functions that require an int)
                let val: f64 = (&**inner).try_into().unwrap();

                *expr = Expr::Const(match ftype {
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
    use super::*;

    use crate::io::qasm::ast::{BinOp, Expr, FuncType, UnOp};

    #[test]
    fn long_expression() {
        // -(-4.0 * sin(2.0) / (2.0 - cos(-1.3)))
        let mut a = Expr::Unary(
            UnOp::Neg,
            Box::new(Expr::Binary(
                BinOp::Div,
                Box::new(Expr::Binary(
                    BinOp::Mul,
                    Box::new(Expr::Unary(UnOp::Neg, Box::new(Expr::Const(4.0)))),
                    Box::new(Expr::Function(FuncType::Sin, Box::new(Expr::Const(2.0)))),
                )),
                Box::new(Expr::Binary(
                    BinOp::Sub,
                    Box::new(Expr::Const(2.0)),
                    Box::new(Expr::Function(
                        FuncType::Cos,
                        Box::new(Expr::Unary(UnOp::Neg, Box::new(Expr::Const(1.3)))),
                    )),
                )),
            )),
        );
        fold_expr(&mut a);
        assert_eq!(
            a,
            Expr::Const(-(-4.0 * 2f64.sin() / (2.0 - (-1.3f64).cos())))
        );
    }

    #[test]
    fn neg_neg() {
        // -(-64.0) = 64.0
        let mut a = Expr::Unary(
            UnOp::Neg,
            Box::new(Expr::Unary(UnOp::Neg, Box::new(Expr::Const(64.0)))),
        );
        fold_expr(&mut a);
        assert_eq!(a, Expr::Const(64.0));
    }

    #[test]
    fn neg_neg_neg() {
        // -(-(-0.1)) = -0.1
        let mut a = Expr::Unary(
            UnOp::Neg,
            Box::new(Expr::Unary(
                UnOp::Neg,
                Box::new(Expr::Unary(UnOp::Neg, Box::new(Expr::Const(0.1)))),
            )),
        );
        fold_expr(&mut a);
        assert_eq!(a, Expr::Const(-0.1));
    }

    #[test]
    fn add() {
        // 2.0 + 4.0
        let mut a = Expr::Binary(
            BinOp::Add,
            Box::new(Expr::Const(2.0)),
            Box::new(Expr::Const(4.0)),
        );
        fold_expr(&mut a);
        assert_eq!(a, Expr::Const(6.0));
    }

    #[test]
    fn add_three() {
        // 0.1 + (0.1 + 0.1)
        let mut a = Expr::Binary(
            BinOp::Add,
            Box::new(Expr::Binary(
                BinOp::Add,
                Box::new(Expr::Const(0.1)),
                Box::new(Expr::Const(0.1)),
            )),
            Box::new(Expr::Const(0.1)),
        );
        fold_expr(&mut a);
        assert_eq!(a, Expr::Const(0.1 + 0.1 + 0.1));
    }

    #[test]
    fn sub() {
        // 2.0 - 4.0
        let mut a = Expr::Binary(
            BinOp::Sub,
            Box::new(Expr::Const(2.0)),
            Box::new(Expr::Const(4.0)),
        );
        fold_expr(&mut a);
        assert_eq!(a, Expr::Const(-2.0));
    }

    #[test]
    fn sub_three() {
        // (0.1 - 0.1) - 0.1
        let mut a = Expr::Binary(
            BinOp::Sub,
            Box::new(Expr::Binary(
                BinOp::Sub,
                Box::new(Expr::Const(0.1)),
                Box::new(Expr::Const(0.1)),
            )),
            Box::new(Expr::Const(0.1)),
        );
        fold_expr(&mut a);
        assert_eq!(a, Expr::Const(-0.1));
    }

    #[test]
    fn multiply() {
        // 2 * 3
        let mut a = Expr::Binary(
            BinOp::Mul,
            Box::new(Expr::Const(2.0)),
            Box::new(Expr::Const(3.0)),
        );
        fold_expr(&mut a);
        assert_eq!(a, Expr::Const(2.0 * 3.0));
    }

    #[test]
    fn multiply_negative() {
        // 3.6 * -2.4
        let mut a = Expr::Binary(
            BinOp::Mul,
            Box::new(Expr::Const(3.6)),
            Box::new(Expr::Const(-2.4)),
        );
        fold_expr(&mut a);
        assert_eq!(a, Expr::Const(3.6 * -2.4));
    }

    #[test]
    fn multiply_both_negative() {
        // -3.0 * -0.5
        let mut a = Expr::Binary(
            BinOp::Mul,
            Box::new(Expr::Const(-3.0)),
            Box::new(Expr::Const(-0.5)),
        );
        fold_expr(&mut a);
        assert_eq!(a, Expr::Const(3.0 * 0.5));
    }

    #[test]
    fn divide() {
        // 30.0 / 4.0
        let mut a = Expr::Binary(
            BinOp::Div,
            Box::new(Expr::Const(30.0)),
            Box::new(Expr::Const(4.0)),
        );
        fold_expr(&mut a);
        assert_eq!(a, Expr::Const(30.0 / 4.0));
    }

    #[test]
    fn divide_negative() {
        // 3.6 / -2.4
        let mut a = Expr::Binary(
            BinOp::Div,
            Box::new(Expr::Const(3.6)),
            Box::new(Expr::Const(-2.4)),
        );
        fold_expr(&mut a);
        assert_eq!(a, Expr::Const(3.6 / -2.4));
    }

    #[test]
    fn divide_both_negative() {
        // -3.0 / -0.5
        let mut a = Expr::Binary(
            BinOp::Div,
            Box::new(Expr::Const(-3.0)),
            Box::new(Expr::Const(-0.5)),
        );
        fold_expr(&mut a);
        assert_eq!(a, Expr::Const(3.0 / 0.5));
    }

    #[test]
    fn power_fractional_base() {
        // 1.5 ^ 2.0
        let mut a = Expr::Binary(
            BinOp::Power,
            Box::new(Expr::Const(1.5)),
            Box::new(Expr::Const(2.0)),
        );
        fold_expr(&mut a);
        assert_eq!(a, Expr::Const(2.25));
    }

    #[test]
    fn single_power() {
        // 5.0 ^ 2.0
        let mut a = Expr::Binary(
            BinOp::Power,
            Box::new(Expr::Const(5.0)),
            Box::new(Expr::Const(2.0)),
        );
        fold_expr(&mut a);
        assert_eq!(a, Expr::Const(25.0));
    }

    #[test]
    fn single_sqrt() {
        // sqrt(4.0)
        let mut a = Expr::Function(FuncType::Sqrt, Box::new(Expr::Const(4.0)));
        fold_expr(&mut a);
        assert_eq!(a, Expr::Const(4f64.sqrt()));
    }

    #[test]
    fn single_cos() {
        // cos(2.3)
        let mut a = Expr::Function(FuncType::Cos, Box::new(Expr::Const(2.3)));
        fold_expr(&mut a);
        assert_eq!(a, Expr::Const(2.3f64.cos()));
    }
}
