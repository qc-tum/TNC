use std::{
    fmt::{Debug, Display},
    ops,
};

use float_cmp::approx_eq;
use itertools::join;

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
/// A unary operator type.
pub enum UnOp {
    Neg,
}

impl Display for UnOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let symbol = match self {
            Self::Neg => "-",
        };
        write!(f, "{symbol}")
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
/// A binary operator type.
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    BitXor,
}

impl Display for BinOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let symbol = match self {
            Self::Add => "+",
            Self::Sub => "-",
            Self::Mul => "*",
            Self::Div => "/",
            Self::BitXor => "^",
        };
        write!(f, "{symbol}")
    }
}

impl BinOp {
    pub const fn get_precedence(self) -> u32 {
        match self {
            Self::Sub => 2,
            Self::Add => 3,
            Self::Mul | Self::Div => 1,
            Self::BitXor => 4,
        }
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
/// A function type.
pub enum FuncType {
    Sin,
    Cos,
    Tan,
    Exp,
    Ln,
    Sqrt,
}

impl Display for FuncType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let symbol = match self {
            Self::Sin => "sin",
            Self::Cos => "cos",
            Self::Tan => "tan",
            Self::Exp => "exp",
            Self::Ln => "ln",
            Self::Sqrt => "sqrt",
        };
        write!(f, "{symbol}")
    }
}

#[derive(Debug, Clone)]
/// A mathematical expression.
pub enum Expr {
    Int(i32),
    Float(f64),
    Variable(String),
    Unary(UnOp, Box<Expr>),
    Binary(BinOp, Box<Expr>, Box<Expr>),
    Function(FuncType, Box<Expr>),
}

impl Display for Expr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Int(x) => write!(f, "{x}"),
            Self::Float(x) => write!(f, "{x}"),
            Self::Variable(x) => write!(f, "{x}"),
            Self::Unary(op, inner) => {
                let need_parens = matches!(**inner, Self::Binary(_, _, _));
                if need_parens {
                    write!(f, "{op}({inner})")
                } else {
                    write!(f, "{op}{inner}")
                }
            }
            Self::Binary(op, lhs, rhs) => {
                let pre = op.get_precedence();
                let lhs_need_parens = if let Self::Binary(opl, _, _) = **lhs {
                    opl.get_precedence() > pre
                } else {
                    false
                };
                let rhs_need_parens = if let Self::Binary(opr, _, _) = **rhs {
                    opr.get_precedence() > pre
                } else {
                    false
                };

                if lhs_need_parens {
                    write!(f, "({lhs})")?;
                } else {
                    write!(f, "{lhs}")?;
                }
                write!(f, " {op} ")?;
                if rhs_need_parens {
                    write!(f, "({rhs})")?;
                } else {
                    write!(f, "{rhs}")?;
                }
                Ok(())
            }
            Self::Function(kind, inner) => write!(f, "{kind}({inner})"),
        }
    }
}

impl PartialEq for Expr {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Int(l0), Self::Int(r0)) => l0 == r0,
            (Self::Float(l0), Self::Float(r0)) => approx_eq!(f64, *l0, *r0),
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
    // Default needed for take()
    fn default() -> Self {
        Self::Int(-1)
    }
}

impl Expr {
    /// Returns whether the expression is a literal.
    pub const fn is_const(&self) -> bool {
        matches!(self, Self::Int(_) | Self::Float(_))
    }
}

impl TryFrom<&Expr> for f64 {
    type Error = ();

    fn try_from(value: &Expr) -> Result<Self, Self::Error> {
        match value {
            Expr::Int(x) => Ok((*x).into()),
            Expr::Float(x) => Ok(*x),
            _ => Err(()),
        }
    }
}

impl ops::Add<&Expr> for &Expr {
    type Output = Expr;

    fn add(self, rhs: &Expr) -> Self::Output {
        if let (Expr::Int(a), Expr::Int(b)) = (self, rhs) {
            Expr::Int(a + b)
        } else {
            let a: f64 = self.try_into().unwrap();
            let b: f64 = rhs.try_into().unwrap();
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
            let a: f64 = self.try_into().unwrap();
            let b: f64 = rhs.try_into().unwrap();
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
            let a: f64 = self.try_into().unwrap();
            let b: f64 = rhs.try_into().unwrap();
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
            let a: f64 = self.try_into().unwrap();
            let b: f64 = rhs.try_into().unwrap();
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

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
/// A name with optionally a designator, e.g. `a` or `q[5]`.
pub struct Argument(pub String, pub Option<u32>);

impl Display for Argument {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)?;
        if let Some(designator) = self.1 {
            write!(f, "[{designator}]")?;
        }
        Ok(())
    }
}

#[derive(Debug, Default, PartialEq)]
/// Declaration of a gate. The body can be None in case of opaque gates.
pub struct GateDeclarationData {
    pub name: String,
    pub params: Vec<String>,
    pub qubits: Vec<String>,
    pub body: Option<Vec<Statement>>,
}

impl Display for GateDeclarationData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.body.is_some() {
            write!(f, "gate")?;
        } else {
            write!(f, "opaque")?;
        }
        write!(f, " {}", self.name)?;

        // Write params
        if !self.params.is_empty() {
            write!(f, "({})", join(&self.params, ", "))?;
        }

        // Write qubits
        write!(f, " {}", join(&self.qubits, ", "))?;

        // Write body
        if let Some(body) = &self.body {
            writeln!(f, " {{")?;
            for statement in body.iter() {
                writeln!(f, "    {statement}")?;
            }
            write!(f, "}}")
        } else {
            write!(f, ";")
        }
    }
}

#[derive(Debug, Default, PartialEq)]
/// Call of a gate.
pub struct GateCallData {
    pub name: String,
    pub args: Vec<Expr>,
    pub qargs: Vec<Argument>,
}

impl Display for GateCallData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name)?;

        // Write args
        if !self.args.is_empty() {
            write!(f, "({})", join(&self.args, ", "))?;
        }

        // Write qargs
        write!(f, " {};", join(&self.qargs, ", "))
    }
}

impl GateCallData {
    /// Returns whether the called gate is builtin, i.e. a `U` or `CX` gate.
    pub fn is_builtin(&self) -> bool {
        self.name == "U" || self.name == "CX"
    }
}

#[derive(Debug, PartialEq)]
/// A QASM2 statement.
pub enum Statement {
    Declaration {
        is_quantum: bool,
        name: String,
        count: u32,
    },
    GateDeclaration(GateDeclarationData),
    GateCall(GateCallData),
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

impl Display for Statement {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Declaration {
                is_quantum,
                name,
                count,
            } => {
                let kind = if *is_quantum { "qreg" } else { "creg" };
                write!(f, "{kind} {name}[{count}];")
            }
            Self::GateDeclaration(data) => write!(f, "{data}"),
            Self::GateCall(data) => write!(f, "{data}"),
            Self::Measurement { src, dest } => write!(f, "measure {src} -> {dest};"),
            Self::Reset { dest } => write!(f, "reset {dest};"),
            Self::IfStatement {
                cond_name,
                condition,
                body,
            } => write!(f, "if {cond_name} = {condition} {body}"),
            Self::Barrier(qargs) => {
                write!(f, "barrier {};", join(qargs, ", "))
            }
        }
    }
}

impl Statement {
    /// Creates a new gate declaration statement.
    pub fn gate_declaration<S>(
        name: S,
        params: Vec<String>,
        qubits: Vec<String>,
        body: Option<Vec<Self>>,
    ) -> Self
    where
        S: Into<String>,
    {
        Self::GateDeclaration(GateDeclarationData {
            name: name.into(),
            params,
            qubits,
            body,
        })
    }

    /// Creates a new gate call statement.
    pub fn gate_call<S>(name: S, args: Vec<Expr>, qargs: Vec<Argument>) -> Self
    where
        S: Into<String>,
    {
        Self::GateCall(GateCallData {
            name: name.into(),
            args,
            qargs,
        })
    }
}

#[derive(Debug, PartialEq)]
/// A QASM2 program.
pub struct Program {
    pub statements: Vec<Statement>,
}

impl Display for Program {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "OPENQASM 2.0;")?;
        for statement in &self.statements {
            writeln!(f, "{statement}")?;
        }
        Ok(())
    }
}

// TODO: the visitor is not really useful at the moment.
// If all enum values had structs inside, we could have visit methods for those.
pub trait Visitor {
    fn visit_program(&mut self, program: &mut Program) {
        for statement in &mut program.statements {
            self.visit_statement(statement);
        }
    }

    fn visit_statement(&mut self, statement: &mut Statement) {
        match statement {
            Statement::GateDeclaration(data) => {
                if let Some(body) = data.body.as_mut() {
                    for statement in body.iter_mut() {
                        self.visit_statement(statement);
                    }
                }
            }
            Statement::GateCall(data) => {
                for expr in &mut data.args {
                    self.visit_expression(expr);
                }
            }
            Statement::IfStatement {
                cond_name: _,
                condition: _,
                body,
            } => self.visit_statement(body),
            _ => (),
        }
    }

    fn visit_expression(&mut self, _expression: &mut Expr) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_program() {
        let program = Program {
            statements: vec![
                Statement::Declaration {
                    is_quantum: true,
                    name: String::from("q"),
                    count: 2,
                },
                Statement::Declaration {
                    is_quantum: false,
                    name: String::from("c"),
                    count: 1,
                },
                Statement::gate_declaration(
                    "x",
                    vec![String::from("a"), String::from("b")],
                    vec![String::from("q")],
                    Some(vec![Statement::gate_call(
                        "U",
                        vec![
                            Expr::Binary(
                                BinOp::Add,
                                Box::new(Expr::Variable(String::from("a"))),
                                Box::new(Expr::Variable(String::from("b"))),
                            ),
                            Expr::Int(0),
                            Expr::Variable(String::from("b")),
                        ],
                        vec![Argument(String::from("q"), None)],
                    )]),
                ),
                Statement::gate_declaration(
                    "y",
                    vec![String::from("a")],
                    vec![String::from("q1"), String::from("q2")],
                    Some(vec![
                        Statement::gate_call(
                            "x",
                            vec![Expr::Variable(String::from("a")), Expr::Int(0)],
                            vec![Argument(String::from("q2"), None)],
                        ),
                        Statement::gate_call(
                            "U",
                            vec![Expr::Int(1), Expr::Int(2), Expr::Int(3)],
                            vec![Argument(String::from("q1"), None)],
                        ),
                        Statement::gate_call(
                            "x",
                            vec![Expr::Int(1), Expr::Variable(String::from("a"))],
                            vec![Argument(String::from("q1"), None)],
                        ),
                    ]),
                ),
                Statement::gate_call(
                    "y",
                    vec![Expr::Int(2)],
                    vec![
                        Argument(String::from("q"), Some(0)),
                        Argument(String::from("q"), Some(1)),
                    ],
                ),
                Statement::Measurement {
                    src: Argument(String::from("q"), Some(1)),
                    dest: Argument(String::from("c"), Some(0)),
                },
            ],
        };

        let out = format!("{program}");
        assert_eq!(
            out,
            "OPENQASM 2.0;
qreg q[2];
creg c[1];
gate x(a, b) q {
    U(a + b, 0, b) q;
}
gate y(a) q1, q2 {
    x(a, 0) q2;
    U(1, 2, 3) q1;
    x(1, a) q1;
}
y(2) q[0], q[1];
measure q[1] -> c[0];
"
        )
    }

    #[test]
    fn display_expression() {
        let expr = Expr::Unary(
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
        let out = format!("{expr}");
        assert_eq!(out, "-(-4 * sin(2) / (2 - cos(-1.3)))");
    }

    #[test]
    fn display2() {
        let expr = Expr::Binary(
            BinOp::Sub,
            Box::new(Expr::Int(5)),
            Box::new(Expr::Binary(
                BinOp::Add,
                Box::new(Expr::Int(2)),
                Box::new(Expr::Int(3)),
            )),
        );
        println!("{expr}");
    }
}
