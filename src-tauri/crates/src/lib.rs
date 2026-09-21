pub mod contracts;
pub mod engine;

pub use contracts::*;
pub use engine::*;

pub fn factory_version() -> &'static str {
    "v1.0.0-BASE-JEV-FACTORY"
}
