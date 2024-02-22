/// Given the module path and name of a test function, returns the name as it is
/// used by cargo test.
///
/// # Examples
/// ```ignore
/// assert_eq!(make_full_test_name("mycrate", "my_test"), "my_test");
/// assert_eq!(make_full_test_name("mycrate::foo", "my_test"), "foo::my_test");
/// assert_eq!(make_full_test_name("mycrate::foo::bar", "my_test"), "foo::bar::my_test");
/// ```
pub(crate) fn make_full_test_name(module_path: &str, test_name: &str) -> String {
    if let Some(idx) = module_path.find("::") {
        // Not in the root module, remove the root name and concat test name
        module_path[idx + 2..].to_string() + "::" + test_name
    } else {
        // In the root module, only use test name
        test_name.to_string()
    }
}

/// Runs a test using `mpirun` with 4 processes. The test must be passed with as full
/// name, e.g., `foo::bar::my_test`.
pub(crate) fn run_mpi_test(test_full_name: &str, processes: usize) {
    let mut command = std::process::Command::new("mpirun");
    command
        .arg("-n")
        .arg(processes.to_string())
        .arg("--allow-run-as-root")
        .arg("cargo")
        .arg("test")
        .arg(test_full_name)
        .arg("--")
        .arg("--ignored")
        .arg("--exact");

    let output = command.status().expect("failed to execute process");
    assert!(output.success());
}

#[macro_export]
macro_rules! mpi_test {
    ($processes:expr, fn $name:ident $_:tt $body:block) => {
        paste::paste! {
            #[test]
            fn $name() {
                let full_path = module_path!();
                let test_name = concat!(stringify!($name), "_internal");
                let exact_name = $crate::mpi::testing::make_full_test_name(full_path, test_name);
                $crate::mpi::testing::run_mpi_test(&exact_name, $processes);
            }

            #[test]
            #[ignore]
            fn [<$name _internal>]() $body
        }
    };
}

#[cfg(test)]
mod tests {
    use super::make_full_test_name;

    #[test]
    fn test_make_test_name() {
        assert_eq!(make_full_test_name("mycrate", "my_test"), "my_test");
        assert_eq!(
            make_full_test_name("mycrate::foo", "my_test"),
            "foo::my_test"
        );
        assert_eq!(
            make_full_test_name("mylongercrate::foo::bar", "my_test"),
            "foo::bar::my_test"
        );
    }
}
