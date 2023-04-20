macro_rules! cast {
    ($target: expr, $pat: path) => {{
        if let $pat(a) = $target {
            a
        } else {
            panic!("Could not cast to {}", stringify!($pat));
        }
    }};
}

pub(crate) use cast;