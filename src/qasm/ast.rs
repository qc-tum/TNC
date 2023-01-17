#[derive(Debug)]
pub enum UnOperator {
    Neg,
}

#[derive(Debug)]
pub enum BinOperator {
    Add,
    Sub,
    Mul,
    Div,
    BitXor,
}

#[derive(Debug)]
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
    Literal(f64),
    Variable(String),
    UnaryExpr(UnOperator, Box<Expr>),
    BinaryExpr(BinOperator, Box<Expr>, Box<Expr>),
    Function(FuncType, Box<Expr>),
}

#[derive(Debug)]
pub struct Argument(pub String, pub Option<u32>);

#[derive(Debug)]
pub struct GCall {
    pub name: String,
    pub args: Vec<Box<Expr>>,
    pub qargs: Vec<Argument>,
}

#[derive(Debug)]
pub enum QOperation {
    GateCall(Box<GCall>),
    Measurement { src: Argument, dest: Argument },
    Reset { dest: Argument },
}

#[derive(Debug)]
pub enum BodyStatement {
    GateCall(Box<GCall>),
    Barrier(Vec<String>),
}

#[derive(Debug)]
pub enum Statement {
    Include(String),
    Declaration {
        is_quantum: bool,
        name: String,
        count: u32,
    },
    GateDeclaration {
        name: String,
        params: Vec<String>,
        qubits: Vec<String>,
        body: Option<Vec<Box<BodyStatement>>>,
    },
    QuantumOperation(Box<QOperation>),
    IfStatement {
        cond_name: String,
        condition: u32,
        body: Box<QOperation>,
    },
    Barrier(Vec<Argument>),
}

#[derive(Debug)]
pub struct Program {
    pub statements: Vec<Box<Statement>>,
}