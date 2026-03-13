//! Doctests for markdown files in the project.
//!
//! This allows us to test e.g. code examples in the README to ensure they stay up-to-date.

#![allow(dead_code, reason = "Doctests need to be placed on some stub object.")]

#[cfg_attr(
    doctest,
    doc = include_str!("../../README.md")
)]
pub struct ReadmeDoctests;
