# Contribution

Thank you for your interest in contributing to this project!

If you encountered a bug, have an idea for a feature, have a question, need help, ... feel free to open an issue and describe your problem.

If you want to contribute code through a PR, here are some guidelines:

## Guidelines for code contributions
- For larger contributions, open an issue first so we can discuss whether the change is in scope of this project
- Make one logical change per commit and write meaningful commit messages
  - If your commit message requires a bullet point list, it's a strong sign that multiple commits would be better
  - When fixing bugs, be sure to describe in the extended part of your commit message: a) what the bug was, b) how it affected users, c) how your change solved it
- Document your code and add tests for it
- Always format your files with `cargo fmt` (best to set up a commit hook or configure your editor to do it on every save)
- If possible, each commit should at least compile without errors
- A PR should only contain one logical change. If you want to add multiple features, open multiple PRs
- The CI must pass for PRs to be merged

## Getting started
- Clone the porject
- Install the required dependencies as described in the README
- Run `cargo build` to ensure everything works
- Do your changes and check with `cargo test` that the tests still pass

## Licensing
Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed under the terms of both the Apache License, Version 2.0 and the MIT license without any additional terms or conditions.
