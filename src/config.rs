use eyre::Result;
use serde::{Deserialize, Serialize};
use std::{fs, path::Path};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub max_batch_size: usize,
    pub max_wait_time_ms: u32,
    pub debug: bool,
}

impl Default for Config {
    fn default() -> Self {

        Self {
            max_batch_size: 32,
            max_wait_time_ms: 1000,
            debug: false,
        }
    }
}

impl std::fmt::Display for Config {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "max_batch_size: {}, max_wait_time_ms: {}, debug: {}", self.max_batch_size, self.max_wait_time_ms, self.debug)
    }
}

impl Config {
    pub fn load(path: &str, debug: bool) -> Result<Self> {
        if !Path::new(path).exists() {
            let mut default_config = Self::default();
            default_config.debug = debug;
            let toml_content = toml::to_string_pretty(&default_config)?;
            fs::write(path, toml_content)?;
            tracing::info!("Created default configuration file at {}", path);
            return Ok(default_config);
        }

        let content = fs::read_to_string(path)?;
        let mut config: Config = toml::from_str(&content)?;
        config.debug = debug;

        config.validate()?;

        Ok(config)
    }

    fn validate(&self) -> Result<()> {
        if self.max_batch_size == 0 {
            return Err(eyre::eyre!("max_batch_size must be greater than 0"));
        }

        if self.max_batch_size > 1000 {
            return Err(eyre::eyre!("max_batch_size too large (max: 1000)"));
        }

        if self.max_wait_time_ms == 0 {
            return Err(eyre::eyre!("max_wait_time_ms must be greater than 0"));
        }

        if self.max_wait_time_ms > 10000 {
            return Err(eyre::eyre!("max_wait_time_ms too large (max: 10000ms)"));
        }

        Ok(())
    }
}
