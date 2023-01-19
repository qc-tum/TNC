use std::ops;

#[derive(Debug, PartialEq, Eq)]
pub enum UnOp {
    Neg,
}

#[derive(Debug, PartialEq, Eq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    BitXor,
}

#[derive(Debug, PartialEq, Eq)]
pub enum FuncType {
    Sin,
    Cos,
    Tan,
    Exp,
    Ln,
    Sqrt,
}

#[derive(Debug)]
pub enum Expr {
    Int(i32),
    Float(f64),
    Variable(String),
    Unary(UnOp, Box<Expr>),
    Binary(BinOp, Box<Expr>, Box<Expr>),
    Function(FuncType, Box<Expr>),
}

impl PartialEq for Expr {
    fn eq(&self, other: &Self) -> bool {
        // Warning: Compares floats directly!
        match (self, other) {
            (Self::Int(l0), Self::Int(r0)) => l0 == r0,
            (Self::Float(l0), Self::Float(r0)) => l0 == r0,
            (Self::Variable(l0), Self::Variable(r0)) => l0 == r0,
            (Self::Unary(l0, l1), Self::Unary(r0, r1)) => l0 == r0 && l1 == r1,
            (Self::Binary(l0, l1, l2), Self::Binary(r0, r1, r2)) => {
                l0 == r0 && l1 == r1 && l2 == r2
            }
            (Self::Function(l0, l1), Self::Function(r0, r1)) => l0 == r0 && l1 == r1,
            _ => false,
        }
    }
}

impl Default for Expr {
    fn default() -> Self {
        Self::Int(-1)
    }
}

impl Expr {
    pub const fn is_const(&self) -> bool {
        matches!(self, Self::Int(_) | Self::Float(_))
    }
}

impl From<&Expr> for f64 {
    fn from(value: &Expr) -> Self {
        match value {
            Expr::Int(x) => (*x).into(),
            Expr::Float(x) => *x,
            _ => panic!("Cannot get value of non-literal expression"),
        }
    }
}

impl ops::Add<&Expr> for &Expr {
    type Output = Expr;

    fn add(self, rhs: &Expr) -> Self::Output {
        if let (Expr::Int(a), Expr::Int(b)) = (self, rhs) {
            Expr::Int(a + b)
        } else {
            let a: f64 = self.into();
            let b: f64 = rhs.into();
            Expr::Float(a + b)
        }
    }
}

impl ops::Sub<&Expr> for &Expr {
    type Output = Expr;

    fn sub(self, rhs: &Expr) -> Self::Output {
        if let (Expr::Int(a), Expr::Int(b)) = (self, rhs) {
            Expr::Int(a - b)
        } else {
            let a: f64 = self.into();
            let b: f64 = rhs.into();
            Expr::Float(a - b)
        }
    }
}

impl ops::Mul<&Expr> for &Expr {
    type Output = Expr;

    fn mul(self, rhs: &Expr) -> Self::Output {
        if let (Expr::Int(a), Expr::Int(b)) = (self, rhs) {
            Expr::Int(a * b)
        } else {
            let a: f64 = self.into();
            let b: f64 = rhs.into();
            Expr::Float(a * b)
        }
    }
}

impl ops::Div<&Expr> for &Expr {
    type Output = Expr;

    fn div(self, rhs: &Expr) -> Self::Output {
        if let (Expr::Int(a), Expr::Int(b)) = (self, rhs) {
            Expr::Int(a / b)
        } else {
            let a: f64 = self.into();
            let b: f64 = rhs.into();
            Expr::Float(a / b)
        }
    }
}

impl ops::BitXor<&Expr> for &Expr {
    type Output = Expr;

    fn bitxor(self, rhs: &Expr) -> Self::Output {
        if let (Expr::Int(a), Expr::Int(b)) = (self, rhs) {
            Expr::Int(a ^ b)
        } else {
            panic!("Cannot apply bitxor on non-int types");
        }
    }
}

#[derive(Debug)]
pub struct Argument(pub String, pub Option<u32>);

#[derive(Debug)]
pub enum Statement {
    Declaration {
        is_quantum: bool,
        name: String,
        count: u32,
    },
    GateDeclaration {
        name: String,
        params: Vec<String>,
        qubits: Vec<String>,
        body: Option<Vec<Statement>>,
    },
    GateCall {
        name: String,
        args: Vec<Expr>,
        qargs: Vec<Argument>,
    },
    Measurement {
        src: Argument,
        dest: Argument,
    },
    Reset {
        dest: Argument,
    },
    IfStatement {
        cond_name: String,
        condition: u32,
        body: Box<Statement>,
    },
    Barrier(Vec<Argument>),
}

#[derive(Debug)]
pub struct Program {
    pub statements: Vec<Statement>,
}

pub trait Visitor {
    fn visit_program(&mut self, program: &mut Program);
    fn visit_statement(&mut self, statement: &mut Statement);
    fn visit_expression(&mut self, expression: &mut Expr);
}
