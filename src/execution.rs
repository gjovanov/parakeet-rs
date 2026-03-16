use crate::error::Result;
use ort::session::builder::SessionBuilder;

// Hardware acceleration options. CPU is default and most reliable.
// GPU providers (CUDA, TensorRT, ROCm) offer 5-10x speedup but require specific hardware.
// All GPU providers automatically fall back to CPU if they fail.
//
// Note: CoreML currently fails with this model due to unsupported operations.
// WebGPU is experimental and may produce incorrect results.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ExecutionProvider {
    #[default]
    Cpu,
    #[cfg(feature = "cuda")]
    Cuda,
    #[cfg(feature = "tensorrt")]
    TensorRT,
    #[cfg(feature = "coreml")]
    CoreML,
    #[cfg(feature = "directml")]
    DirectML,
    #[cfg(feature = "rocm")]
    ROCm,
    #[cfg(feature = "openvino")]
    OpenVINO,
    #[cfg(feature = "webgpu")]
    WebGPU,
}

impl ExecutionProvider {
    /// Detect the best available execution provider based on environment and compiled features.
    ///
    /// Checks `USE_GPU` environment variable:
    /// - `USE_GPU=true` or `USE_GPU=1`: Attempts to use GPU (CUDA > TensorRT > CPU)
    /// - `USE_GPU=cuda`: Specifically request CUDA
    /// - `USE_GPU=tensorrt`: Specifically request TensorRT
    /// - Otherwise: Uses CPU
    ///
    /// Returns CPU if the requested provider is not compiled in.
    pub fn from_env() -> Self {
        let use_gpu = std::env::var("USE_GPU").unwrap_or_default().to_lowercase();

        match use_gpu.as_str() {
            "true" | "1" | "yes" => {
                // Try CUDA first, then TensorRT, then CPU
                #[cfg(feature = "cuda")]
                {
                    eprintln!("[GPU] USE_GPU=true, using CUDA execution provider");
                    return Self::Cuda;
                }
                #[cfg(feature = "tensorrt")]
                {
                    eprintln!("[GPU] USE_GPU=true, using TensorRT execution provider");
                    return Self::TensorRT;
                }
                #[cfg(not(any(feature = "cuda", feature = "tensorrt")))]
                {
                    eprintln!("[GPU] USE_GPU=true but no GPU features compiled, falling back to CPU");
                    Self::Cpu
                }
            }
            "cuda" => {
                #[cfg(feature = "cuda")]
                {
                    eprintln!("[GPU] USE_GPU=cuda, using CUDA execution provider");
                    return Self::Cuda;
                }
                #[cfg(not(feature = "cuda"))]
                {
                    eprintln!("[GPU] USE_GPU=cuda but CUDA feature not compiled, falling back to CPU");
                    Self::Cpu
                }
            }
            "tensorrt" => {
                #[cfg(feature = "tensorrt")]
                {
                    eprintln!("[GPU] USE_GPU=tensorrt, using TensorRT execution provider");
                    return Self::TensorRT;
                }
                #[cfg(not(feature = "tensorrt"))]
                {
                    eprintln!("[GPU] USE_GPU=tensorrt but TensorRT feature not compiled, falling back to CPU");
                    Self::Cpu
                }
            }
            "rocm" => {
                #[cfg(feature = "rocm")]
                {
                    eprintln!("[GPU] USE_GPU=rocm, using ROCm execution provider");
                    return Self::ROCm;
                }
                #[cfg(not(feature = "rocm"))]
                {
                    eprintln!("[GPU] USE_GPU=rocm but ROCm feature not compiled, falling back to CPU");
                    Self::Cpu
                }
            }
            _ => {
                if !use_gpu.is_empty() {
                    eprintln!("[GPU] Unknown USE_GPU value '{}', using CPU", use_gpu);
                }
                Self::Cpu
            }
        }
    }

    /// Returns true if this is a GPU-accelerated provider
    pub fn is_gpu(&self) -> bool {
        !matches!(self, Self::Cpu)
    }

    /// Returns a human-readable name for the provider
    pub fn name(&self) -> &'static str {
        match self {
            Self::Cpu => "CPU",
            #[cfg(feature = "cuda")]
            Self::Cuda => "CUDA",
            #[cfg(feature = "tensorrt")]
            Self::TensorRT => "TensorRT",
            #[cfg(feature = "coreml")]
            Self::CoreML => "CoreML",
            #[cfg(feature = "directml")]
            Self::DirectML => "DirectML",
            #[cfg(feature = "rocm")]
            Self::ROCm => "ROCm",
            #[cfg(feature = "openvino")]
            Self::OpenVINO => "OpenVINO",
            #[cfg(feature = "webgpu")]
            Self::WebGPU => "WebGPU",
        }
    }
}

// ---------------------------------------------------------------------------
// Phase 5: GPU optimization types for RTX 5090 (Blackwell, 24 GB GDDR7)
// ---------------------------------------------------------------------------

/// Default VRAM budget: 20 GB out of 24 GB total (reserves 4 GB for OS/display).
/// Target hardware: RTX 5090 Laptop (24 GB GDDR7, Blackwell arch),
/// AMD Ryzen 9 9955HX3D (16 cores, 3D V-Cache).
const DEFAULT_VRAM_BUDGET_GB: f64 = 20.0;
const GB: u64 = 1_073_741_824; // 1 GiB in bytes

/// Describes the computational role a model plays in the pipeline.
///
/// Each role carries different optimization trade-offs regarding precision,
/// latency, memory allocation strategy, and kernel selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelRole {
    /// Audio/feature encoder -- the heaviest compute kernel, runs on fixed-shape
    /// input tensors.  Optimized for maximum throughput via TensorRT with FP16+INT8,
    /// exhaustive cuDNN convolution search, and CUDA graph capture.
    /// Engine caching avoids costly re-compilation across restarts.
    Encoder,

    /// Autoregressive text decoder -- operates on variable-length sequences that
    /// grow token-by-token.  TensorRT is avoided because dynamic shapes cause
    /// frequent engine rebuilds.  Uses CUDA EP with heuristic cuDNN search and
    /// next-power-of-two arena extension to reduce re-allocation overhead.
    Decoder,

    /// Lightweight auxiliary models (VAD, end-of-utterance, speaker embedding).
    /// Optimized for fast startup and low overhead: CUDA graphs enabled for
    /// fixed-shape inference, default cuDNN search (no benchmarking delay),
    /// reduced intra-op thread count.
    Utility,

    /// Text formatting / punctuation-restoration model -- typically autoregressive
    /// but with smaller footprint.  Runs on CUDA without CUDA graphs (incompatible
    /// with autoregressive control flow).  Uses auxiliary streams to overlap
    /// compute with data transfers.
    Formatter,
}

impl std::fmt::Display for ModelRole {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Encoder => write!(f, "Encoder"),
            Self::Decoder => write!(f, "Decoder"),
            Self::Utility => write!(f, "Utility"),
            Self::Formatter => write!(f, "Formatter"),
        }
    }
}

/// GPU optimization aggressiveness, controlled via `GPU_OPTIMIZATION` env var.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuOptimizationLevel {
    /// GPU optimizations disabled -- plain defaults, useful for debugging.
    Disabled,
    /// Conservative: enable basic tuning (device id, memory limit, TF32) but skip
    /// TensorRT, CUDA graphs, and exhaustive search.
    Conservative,
    /// Aggressive (default): full role-based optimization including TensorRT engine
    /// caching, INT8 quantization, CUDA graphs, and exhaustive cuDNN search where
    /// appropriate.
    Aggressive,
}

impl GpuOptimizationLevel {
    fn from_env_value(val: &str) -> Self {
        match val {
            "0" | "disabled" | "off" | "false" => Self::Disabled,
            "1" | "conservative" => Self::Conservative,
            _ => Self::Aggressive, // "2", "aggressive", or any other value
        }
    }
}

impl Default for GpuOptimizationLevel {
    fn default() -> Self {
        Self::Aggressive
    }
}

#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub execution_provider: ExecutionProvider,
    pub intra_threads: usize,
    pub inter_threads: usize,

    // -- Phase 5: GPU optimization fields (all Option for backward compat) --

    /// The computational role this model plays, driving role-specific EP tuning.
    pub model_role: Option<ModelRole>,
    /// CUDA device ordinal (default: 0).
    pub device_id: Option<i32>,
    /// VRAM budget in bytes for the CUDA memory arena.
    pub gpu_mem_limit_bytes: Option<u64>,
    /// Enable TensorRT FP16 precision.
    pub trt_fp16_enable: Option<bool>,
    /// Enable TensorRT INT8 quantization.
    pub trt_int8_enable: Option<bool>,
    /// Enable TensorRT engine caching to avoid re-compilation.
    pub trt_engine_cache_enable: Option<bool>,
    /// Filesystem path for cached TensorRT engines.
    pub trt_engine_cache_path: Option<String>,
    /// Maximum TensorRT workspace size in bytes.
    pub trt_max_workspace_size: Option<usize>,
    /// TensorRT builder optimization level (0-5, higher = slower build but faster inference).
    pub trt_builder_optimization_level: Option<u8>,
    /// CUDA arena extension strategy: "next_power_of_two" or "same_as_requested".
    pub cuda_arena_extend_strategy: Option<String>,
    /// cuDNN convolution algorithm search mode: "exhaustive", "heuristic", or "default".
    pub cuda_cudnn_conv_algo_search: Option<String>,
    /// Enable CUDA graph capture for reduced kernel-launch overhead.
    /// Incompatible with dynamic shapes and control-flow ops.
    pub cuda_enable_cuda_graph: Option<bool>,
    /// Enable TF32 for faster matmul/conv on Ampere+ (Blackwell included).
    pub cuda_enable_tf32: Option<bool>,
    /// GPU optimization level governing how aggressively role-based tuning is applied.
    pub gpu_optimization_level: Option<GpuOptimizationLevel>,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            execution_provider: ExecutionProvider::default(),
            intra_threads: 4,
            inter_threads: 1,
            model_role: None,
            device_id: None,
            gpu_mem_limit_bytes: None,
            trt_fp16_enable: None,
            trt_int8_enable: None,
            trt_engine_cache_enable: None,
            trt_engine_cache_path: None,
            trt_max_workspace_size: None,
            trt_builder_optimization_level: None,
            cuda_arena_extend_strategy: None,
            cuda_cudnn_conv_algo_search: None,
            cuda_enable_cuda_graph: None,
            cuda_enable_tf32: None,
            gpu_optimization_level: None,
        }
    }
}

impl ModelConfig {
    pub fn new() -> Self {
        Self::default()
    }

    /// Create ModelConfig from environment variables.
    ///
    /// Checks:
    /// - `USE_GPU`: Selects execution provider (see `ExecutionProvider::from_env()`)
    /// - `INTRA_THREADS`: Number of intra-op threads (default: 4)
    /// - `INTER_THREADS`: Number of inter-op threads (default: 1)
    /// - `CUDA_DEVICE_ID`: GPU device ordinal (default: 0)
    /// - `VRAM_BUDGET_GB`: VRAM budget in gigabytes (default: 20.0 for RTX 5090 Laptop)
    /// - `TRT_CACHE_PATH`: Filesystem path for TensorRT engine cache
    /// - `GPU_OPTIMIZATION`: Optimization level -- 0=disabled, 1=conservative, 2=aggressive (default)
    pub fn from_env() -> Self {
        let provider = ExecutionProvider::from_env();
        let intra_threads = std::env::var("INTRA_THREADS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(4);
        let inter_threads = std::env::var("INTER_THREADS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1);

        let device_id = std::env::var("CUDA_DEVICE_ID")
            .ok()
            .and_then(|s| s.parse::<i32>().ok());

        let gpu_mem_limit_bytes = std::env::var("VRAM_BUDGET_GB")
            .ok()
            .and_then(|s| s.parse::<f64>().ok())
            .map(|gb| (gb * GB as f64) as u64);

        let trt_engine_cache_path = std::env::var("TRT_CACHE_PATH").ok();

        let gpu_optimization_level = std::env::var("GPU_OPTIMIZATION")
            .ok()
            .map(|s| GpuOptimizationLevel::from_env_value(&s.to_lowercase()));

        eprintln!(
            "[Config] Execution provider: {}, intra_threads: {}, inter_threads: {}{}{}{}",
            provider.name(),
            intra_threads,
            inter_threads,
            device_id
                .map(|id| format!(", cuda_device_id: {}", id))
                .unwrap_or_default(),
            gpu_mem_limit_bytes
                .map(|b| format!(", vram_budget: {:.1} GB", b as f64 / GB as f64))
                .unwrap_or_default(),
            gpu_optimization_level
                .map(|l| format!(", gpu_optimization: {:?}", l))
                .unwrap_or_default(),
        );

        Self {
            execution_provider: provider,
            intra_threads,
            inter_threads,
            model_role: None,
            device_id,
            gpu_mem_limit_bytes,
            trt_fp16_enable: None,
            trt_int8_enable: None,
            trt_engine_cache_enable: None,
            trt_engine_cache_path: trt_engine_cache_path,
            trt_max_workspace_size: None,
            trt_builder_optimization_level: None,
            cuda_arena_extend_strategy: None,
            cuda_cudnn_conv_algo_search: None,
            cuda_enable_cuda_graph: None,
            cuda_enable_tf32: None,
            gpu_optimization_level,
        }
    }

    pub fn with_execution_provider(mut self, provider: ExecutionProvider) -> Self {
        self.execution_provider = provider;
        self
    }

    pub fn with_intra_threads(mut self, threads: usize) -> Self {
        self.intra_threads = threads;
        self
    }

    pub fn with_inter_threads(mut self, threads: usize) -> Self {
        self.inter_threads = threads;
        self
    }

    /// Assign a model role to this config. Returns self for chaining.
    pub fn with_role(mut self, role: ModelRole) -> Self {
        self.model_role = Some(role);
        self
    }

    /// Create a role-optimized config by cloning a base config and applying
    /// role-specific GPU tuning.
    ///
    /// The optimization level is read from `base.gpu_optimization_level`
    /// (falling back to `Aggressive`). When `Disabled`, returns the base config
    /// with only the role tag set.
    ///
    /// # Role strategies
    ///
    /// - **Encoder**: TensorRT EP with FP16+INT8, exhaustive cuDNN search,
    ///   CUDA graphs, engine caching to `TRT_CACHE_PATH`, builder opt level 5,
    ///   TF32, 2 GB TRT workspace.
    ///
    /// - **Decoder**: CUDA EP (no TensorRT -- variable sequence lengths cause
    ///   engine rebuilds), heuristic cuDNN search, next-power-of-two arena,
    ///   TF32 enabled.
    ///
    /// - **Utility**: CUDA EP, CUDA graphs enabled, default cuDNN search for fast
    ///   startup, 2 intra-op threads.
    ///
    /// - **Formatter**: CUDA EP, no CUDA graphs (autoregressive), default cuDNN,
    ///   TF32 enabled.
    pub fn for_role(role: ModelRole, base: &ModelConfig) -> Self {
        let opt = base
            .gpu_optimization_level
            .unwrap_or(GpuOptimizationLevel::Aggressive);

        let mut cfg = base.clone();
        cfg.model_role = Some(role);

        if opt == GpuOptimizationLevel::Disabled {
            return cfg;
        }

        // Common GPU defaults for any optimization level
        if cfg.device_id.is_none() {
            cfg.device_id = Some(0);
        }
        if cfg.gpu_mem_limit_bytes.is_none() {
            cfg.gpu_mem_limit_bytes = Some((DEFAULT_VRAM_BUDGET_GB * GB as f64) as u64);
        }
        cfg.cuda_enable_tf32 = Some(true);

        if opt == GpuOptimizationLevel::Conservative {
            // Conservative: basic tuning only, no TRT/CUDA graphs/exhaustive search
            return cfg;
        }

        // Aggressive: full role-based optimization
        match role {
            ModelRole::Encoder => {
                // TensorRT with maximum optimization for fixed-shape encoder
                #[cfg(feature = "tensorrt")]
                {
                    cfg.execution_provider = ExecutionProvider::TensorRT;
                }
                cfg.trt_fp16_enable = Some(true);
                cfg.trt_int8_enable = Some(true);
                cfg.trt_engine_cache_enable = Some(true);
                if cfg.trt_engine_cache_path.is_none() {
                    cfg.trt_engine_cache_path = std::env::var("TRT_CACHE_PATH").ok().or_else(|| {
                        Some("/tmp/parakeet_trt_cache".to_string())
                    });
                }
                cfg.trt_max_workspace_size = Some(2 * GB as usize); // 2 GB
                cfg.trt_builder_optimization_level = Some(5);
                cfg.cuda_cudnn_conv_algo_search = Some("exhaustive".to_string());
                cfg.cuda_enable_cuda_graph = Some(true);
            }
            ModelRole::Decoder => {
                // CUDA only -- TRT engine rebuilds on every new sequence length
                #[cfg(feature = "cuda")]
                {
                    cfg.execution_provider = ExecutionProvider::Cuda;
                }
                cfg.cuda_cudnn_conv_algo_search = Some("heuristic".to_string());
                cfg.cuda_arena_extend_strategy = Some("next_power_of_two".to_string());
                cfg.cuda_enable_cuda_graph = Some(false);
            }
            ModelRole::Utility => {
                // Fast startup, low overhead for small auxiliary models
                #[cfg(feature = "cuda")]
                {
                    cfg.execution_provider = ExecutionProvider::Cuda;
                }
                cfg.cuda_cudnn_conv_algo_search = Some("default".to_string());
                cfg.cuda_enable_cuda_graph = Some(true);
                cfg.intra_threads = 2;
            }
            ModelRole::Formatter => {
                // Autoregressive text model -- no CUDA graphs, default search
                #[cfg(feature = "cuda")]
                {
                    cfg.execution_provider = ExecutionProvider::Cuda;
                }
                cfg.cuda_cudnn_conv_algo_search = Some("default".to_string());
                cfg.cuda_enable_cuda_graph = Some(false);
            }
        }

        cfg
    }

    // -------------------------------------------------------------------
    // Internal helpers for apply_to_session_builder
    // -------------------------------------------------------------------

    /// Resolve the cuDNN convolution algorithm search enum from the string config.
    #[cfg(feature = "cuda")]
    fn resolve_cudnn_conv_algo(
        &self,
    ) -> Option<ort::execution_providers::cuda::ConvAlgorithmSearch> {
        use ort::execution_providers::cuda::ConvAlgorithmSearch;
        self.cuda_cudnn_conv_algo_search.as_deref().map(|s| match s {
            "exhaustive" => ConvAlgorithmSearch::Exhaustive,
            "heuristic" => ConvAlgorithmSearch::Heuristic,
            "default" => ConvAlgorithmSearch::Default,
            _ => ConvAlgorithmSearch::Exhaustive,
        })
    }

    /// Resolve the arena extend strategy enum from the string config.
    #[cfg(feature = "cuda")]
    fn resolve_arena_strategy(&self) -> Option<ort::execution_providers::ArenaExtendStrategy> {
        use ort::execution_providers::ArenaExtendStrategy;
        self.cuda_arena_extend_strategy.as_deref().map(|s| match s {
            "same_as_requested" => ArenaExtendStrategy::SameAsRequested,
            "next_power_of_two" => ArenaExtendStrategy::NextPowerOfTwo,
            _ => ArenaExtendStrategy::NextPowerOfTwo,
        })
    }

    /// Build a configured CUDAExecutionProvider from this config's GPU fields.
    #[cfg(feature = "cuda")]
    fn build_cuda_ep(&self) -> ort::execution_providers::CUDAExecutionProvider {
        let mut ep = ort::execution_providers::CUDAExecutionProvider::default();

        if let Some(id) = self.device_id {
            ep = ep.with_device_id(id);
        }
        if let Some(limit) = self.gpu_mem_limit_bytes {
            ep = ep.with_memory_limit(limit as usize);
        }
        if let Some(algo) = self.resolve_cudnn_conv_algo() {
            ep = ep.with_conv_algorithm_search(algo);
        }
        if let Some(strategy) = self.resolve_arena_strategy() {
            ep = ep.with_arena_extend_strategy(strategy);
        }
        if let Some(enable) = self.cuda_enable_cuda_graph {
            ep = ep.with_cuda_graph(enable);
        }
        if let Some(enable) = self.cuda_enable_tf32 {
            ep = ep.with_tf32(enable);
        }

        ep
    }

    /// Build a configured TensorRTExecutionProvider from this config's TRT fields.
    #[cfg(feature = "tensorrt")]
    fn build_trt_ep(&self) -> ort::execution_providers::TensorRTExecutionProvider {
        let mut ep = ort::execution_providers::TensorRTExecutionProvider::default();

        if let Some(id) = self.device_id {
            ep = ep.with_device_id(id);
        }
        if let Some(enable) = self.trt_fp16_enable {
            ep = ep.with_fp16(enable);
        }
        if let Some(enable) = self.trt_int8_enable {
            ep = ep.with_int8(enable);
        }
        if let Some(enable) = self.trt_engine_cache_enable {
            ep = ep.with_engine_cache(enable);
        }
        if let Some(ref path) = self.trt_engine_cache_path {
            ep = ep.with_engine_cache_path(path);
        }
        if let Some(size) = self.trt_max_workspace_size {
            ep = ep.with_max_workspace_size(size);
        }
        if let Some(level) = self.trt_builder_optimization_level {
            ep = ep.with_builder_optimization_level(level);
        }
        // Enable timing cache alongside engine cache for faster rebuilds
        if self.trt_engine_cache_enable == Some(true) {
            ep = ep.with_timing_cache(true);
            if let Some(ref path) = self.trt_engine_cache_path {
                ep = ep.with_timing_cache_path(path);
            }
        }

        ep
    }

    pub(crate) fn apply_to_session_builder(
        &self,
        builder: SessionBuilder,
    ) -> Result<SessionBuilder> {
        #[cfg(any(
            feature = "cuda",
            feature = "tensorrt",
            feature = "coreml",
            feature = "directml",
            feature = "rocm",
            feature = "openvino",
            feature = "webgpu"
        ))]
        use ort::execution_providers::CPUExecutionProvider;
        use ort::session::builder::GraphOptimizationLevel;

        let mut builder = builder
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(self.intra_threads)?
            .with_inter_threads(self.inter_threads)?;

        builder = match self.execution_provider {
            ExecutionProvider::Cpu => builder,

            #[cfg(feature = "cuda")]
            ExecutionProvider::Cuda => builder.with_execution_providers([
                self.build_cuda_ep().build().error_on_failure(),
                CPUExecutionProvider::default().build(),
            ])?,

            #[cfg(feature = "tensorrt")]
            ExecutionProvider::TensorRT => {
                let mut eps = vec![self.build_trt_ep().build()];
                // Fall back to CUDA with the same GPU tuning when available
                #[cfg(feature = "cuda")]
                {
                    eps.push(self.build_cuda_ep().build());
                }
                eps.push(CPUExecutionProvider::default().build().error_on_failure());
                builder.with_execution_providers(eps)?
            }

            #[cfg(feature = "coreml")]
            ExecutionProvider::CoreML => {
                use ort::execution_providers::coreml::{
                    CoreMLComputeUnits, CoreMLExecutionProvider,
                };
                builder.with_execution_providers([
                    CoreMLExecutionProvider::default()
                        .with_compute_units(CoreMLComputeUnits::CPUAndGPU)
                        .build(),
                    CPUExecutionProvider::default().build().error_on_failure(),
                ])?
            }

            #[cfg(feature = "directml")]
            ExecutionProvider::DirectML => builder.with_execution_providers([
                ort::execution_providers::DirectMLExecutionProvider::default().build(),
                CPUExecutionProvider::default().build().error_on_failure(),
            ])?,

            #[cfg(feature = "rocm")]
            ExecutionProvider::ROCm => builder.with_execution_providers([
                ort::execution_providers::ROCmExecutionProvider::default().build(),
                CPUExecutionProvider::default().build().error_on_failure(),
            ])?,

            #[cfg(feature = "openvino")]
            ExecutionProvider::OpenVINO => builder.with_execution_providers([
                ort::execution_providers::OpenVINOExecutionProvider::default().build(),
                CPUExecutionProvider::default().build().error_on_failure(),
            ])?,

            #[cfg(feature = "webgpu")]
            ExecutionProvider::WebGPU => builder.with_execution_providers([
                ort::execution_providers::WebGPUExecutionProvider::default().build(),
                CPUExecutionProvider::default().build().error_on_failure(),
            ])?,
        };

        Ok(builder)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Backward compatibility: default ModelConfig has no GPU fields set
    // -----------------------------------------------------------------------

    #[test]
    fn default_config_has_no_gpu_fields() {
        let cfg = ModelConfig::default();
        assert_eq!(cfg.intra_threads, 4);
        assert_eq!(cfg.inter_threads, 1);
        assert!(cfg.model_role.is_none());
        assert!(cfg.device_id.is_none());
        assert!(cfg.gpu_mem_limit_bytes.is_none());
        assert!(cfg.trt_fp16_enable.is_none());
        assert!(cfg.trt_int8_enable.is_none());
        assert!(cfg.trt_engine_cache_enable.is_none());
        assert!(cfg.trt_engine_cache_path.is_none());
        assert!(cfg.trt_max_workspace_size.is_none());
        assert!(cfg.trt_builder_optimization_level.is_none());
        assert!(cfg.cuda_arena_extend_strategy.is_none());
        assert!(cfg.cuda_cudnn_conv_algo_search.is_none());
        assert!(cfg.cuda_enable_cuda_graph.is_none());
        assert!(cfg.cuda_enable_tf32.is_none());
        assert!(cfg.gpu_optimization_level.is_none());
    }

    #[test]
    fn new_is_same_as_default() {
        let a = ModelConfig::new();
        let b = ModelConfig::default();
        assert_eq!(a.intra_threads, b.intra_threads);
        assert_eq!(a.inter_threads, b.inter_threads);
        assert!(a.model_role.is_none());
    }

    // -----------------------------------------------------------------------
    // Builder methods
    // -----------------------------------------------------------------------

    #[test]
    fn with_role_sets_model_role() {
        let cfg = ModelConfig::new().with_role(ModelRole::Encoder);
        assert_eq!(cfg.model_role, Some(ModelRole::Encoder));

        let cfg = cfg.with_role(ModelRole::Decoder);
        assert_eq!(cfg.model_role, Some(ModelRole::Decoder));
    }

    #[test]
    fn builder_methods_chain() {
        let cfg = ModelConfig::new()
            .with_intra_threads(8)
            .with_inter_threads(2)
            .with_role(ModelRole::Utility);
        assert_eq!(cfg.intra_threads, 8);
        assert_eq!(cfg.inter_threads, 2);
        assert_eq!(cfg.model_role, Some(ModelRole::Utility));
    }

    // -----------------------------------------------------------------------
    // for_role: Encoder
    // -----------------------------------------------------------------------

    #[test]
    fn for_role_encoder_aggressive() {
        let base = ModelConfig {
            gpu_optimization_level: Some(GpuOptimizationLevel::Aggressive),
            ..ModelConfig::default()
        };
        let cfg = ModelConfig::for_role(ModelRole::Encoder, &base);

        assert_eq!(cfg.model_role, Some(ModelRole::Encoder));
        assert_eq!(cfg.trt_fp16_enable, Some(true));
        assert_eq!(cfg.trt_int8_enable, Some(true));
        assert_eq!(cfg.trt_engine_cache_enable, Some(true));
        assert!(cfg.trt_engine_cache_path.is_some());
        assert_eq!(cfg.trt_max_workspace_size, Some(2 * GB as usize));
        assert_eq!(cfg.trt_builder_optimization_level, Some(5));
        assert_eq!(
            cfg.cuda_cudnn_conv_algo_search.as_deref(),
            Some("exhaustive")
        );
        assert_eq!(cfg.cuda_enable_cuda_graph, Some(true));
        assert_eq!(cfg.cuda_enable_tf32, Some(true));
        assert_eq!(cfg.device_id, Some(0));
        assert!(cfg.gpu_mem_limit_bytes.is_some());
    }

    // -----------------------------------------------------------------------
    // for_role: Decoder
    // -----------------------------------------------------------------------

    #[test]
    fn for_role_decoder_aggressive() {
        let base = ModelConfig {
            gpu_optimization_level: Some(GpuOptimizationLevel::Aggressive),
            ..ModelConfig::default()
        };
        let cfg = ModelConfig::for_role(ModelRole::Decoder, &base);

        assert_eq!(cfg.model_role, Some(ModelRole::Decoder));
        assert_eq!(
            cfg.cuda_cudnn_conv_algo_search.as_deref(),
            Some("heuristic")
        );
        assert_eq!(
            cfg.cuda_arena_extend_strategy.as_deref(),
            Some("next_power_of_two")
        );
        assert_eq!(cfg.cuda_enable_cuda_graph, Some(false));
        assert_eq!(cfg.cuda_enable_tf32, Some(true));
        // Should NOT set TRT fields
        assert!(cfg.trt_fp16_enable.is_none());
        assert!(cfg.trt_int8_enable.is_none());
    }

    // -----------------------------------------------------------------------
    // for_role: Utility
    // -----------------------------------------------------------------------

    #[test]
    fn for_role_utility_aggressive() {
        let base = ModelConfig {
            gpu_optimization_level: Some(GpuOptimizationLevel::Aggressive),
            ..ModelConfig::default()
        };
        let cfg = ModelConfig::for_role(ModelRole::Utility, &base);

        assert_eq!(cfg.model_role, Some(ModelRole::Utility));
        assert_eq!(
            cfg.cuda_cudnn_conv_algo_search.as_deref(),
            Some("default")
        );
        assert_eq!(cfg.cuda_enable_cuda_graph, Some(true));
        assert_eq!(cfg.intra_threads, 2);
        assert_eq!(cfg.cuda_enable_tf32, Some(true));
    }

    // -----------------------------------------------------------------------
    // for_role: Formatter
    // -----------------------------------------------------------------------

    #[test]
    fn for_role_formatter_aggressive() {
        let base = ModelConfig {
            gpu_optimization_level: Some(GpuOptimizationLevel::Aggressive),
            ..ModelConfig::default()
        };
        let cfg = ModelConfig::for_role(ModelRole::Formatter, &base);

        assert_eq!(cfg.model_role, Some(ModelRole::Formatter));
        assert_eq!(
            cfg.cuda_cudnn_conv_algo_search.as_deref(),
            Some("default")
        );
        assert_eq!(cfg.cuda_enable_cuda_graph, Some(false));
        assert_eq!(cfg.cuda_enable_tf32, Some(true));
    }

    // -----------------------------------------------------------------------
    // for_role: Disabled optimization level
    // -----------------------------------------------------------------------

    #[test]
    fn for_role_disabled_returns_base_with_role_tag() {
        let base = ModelConfig {
            gpu_optimization_level: Some(GpuOptimizationLevel::Disabled),
            intra_threads: 8,
            ..ModelConfig::default()
        };
        let cfg = ModelConfig::for_role(ModelRole::Encoder, &base);

        assert_eq!(cfg.model_role, Some(ModelRole::Encoder));
        // No GPU tuning applied
        assert!(cfg.trt_fp16_enable.is_none());
        assert!(cfg.cuda_enable_cuda_graph.is_none());
        assert!(cfg.device_id.is_none());
        assert!(cfg.gpu_mem_limit_bytes.is_none());
        // Base values preserved
        assert_eq!(cfg.intra_threads, 8);
    }

    // -----------------------------------------------------------------------
    // for_role: Conservative optimization level
    // -----------------------------------------------------------------------

    #[test]
    fn for_role_conservative_sets_basics_only() {
        let base = ModelConfig {
            gpu_optimization_level: Some(GpuOptimizationLevel::Conservative),
            ..ModelConfig::default()
        };
        let cfg = ModelConfig::for_role(ModelRole::Encoder, &base);

        assert_eq!(cfg.model_role, Some(ModelRole::Encoder));
        // Conservative sets device_id, mem limit, TF32
        assert_eq!(cfg.device_id, Some(0));
        assert!(cfg.gpu_mem_limit_bytes.is_some());
        assert_eq!(cfg.cuda_enable_tf32, Some(true));
        // But does NOT set aggressive TRT / CUDA graph / exhaustive search
        assert!(cfg.trt_fp16_enable.is_none());
        assert!(cfg.trt_int8_enable.is_none());
        assert!(cfg.cuda_enable_cuda_graph.is_none());
        assert!(cfg.cuda_cudnn_conv_algo_search.is_none());
    }

    // -----------------------------------------------------------------------
    // for_role: base device_id / mem_limit preserved
    // -----------------------------------------------------------------------

    #[test]
    fn for_role_preserves_base_device_id_and_mem() {
        let base = ModelConfig {
            device_id: Some(1),
            gpu_mem_limit_bytes: Some(16 * GB),
            gpu_optimization_level: Some(GpuOptimizationLevel::Aggressive),
            ..ModelConfig::default()
        };
        let cfg = ModelConfig::for_role(ModelRole::Decoder, &base);
        assert_eq!(cfg.device_id, Some(1));
        assert_eq!(cfg.gpu_mem_limit_bytes, Some(16 * GB));
    }

    // -----------------------------------------------------------------------
    // for_role: encoder preserves existing trt_engine_cache_path
    // -----------------------------------------------------------------------

    #[test]
    fn for_role_encoder_preserves_explicit_cache_path() {
        let base = ModelConfig {
            trt_engine_cache_path: Some("/my/cache".to_string()),
            gpu_optimization_level: Some(GpuOptimizationLevel::Aggressive),
            ..ModelConfig::default()
        };
        let cfg = ModelConfig::for_role(ModelRole::Encoder, &base);
        assert_eq!(cfg.trt_engine_cache_path.as_deref(), Some("/my/cache"));
    }

    // -----------------------------------------------------------------------
    // GpuOptimizationLevel parsing
    // -----------------------------------------------------------------------

    #[test]
    fn gpu_optimization_level_from_env_values() {
        assert_eq!(
            GpuOptimizationLevel::from_env_value("0"),
            GpuOptimizationLevel::Disabled
        );
        assert_eq!(
            GpuOptimizationLevel::from_env_value("disabled"),
            GpuOptimizationLevel::Disabled
        );
        assert_eq!(
            GpuOptimizationLevel::from_env_value("off"),
            GpuOptimizationLevel::Disabled
        );
        assert_eq!(
            GpuOptimizationLevel::from_env_value("false"),
            GpuOptimizationLevel::Disabled
        );
        assert_eq!(
            GpuOptimizationLevel::from_env_value("1"),
            GpuOptimizationLevel::Conservative
        );
        assert_eq!(
            GpuOptimizationLevel::from_env_value("conservative"),
            GpuOptimizationLevel::Conservative
        );
        assert_eq!(
            GpuOptimizationLevel::from_env_value("2"),
            GpuOptimizationLevel::Aggressive
        );
        assert_eq!(
            GpuOptimizationLevel::from_env_value("aggressive"),
            GpuOptimizationLevel::Aggressive
        );
        // Unknown falls back to aggressive
        assert_eq!(
            GpuOptimizationLevel::from_env_value("banana"),
            GpuOptimizationLevel::Aggressive
        );
    }

    // -----------------------------------------------------------------------
    // ModelRole Display
    // -----------------------------------------------------------------------

    #[test]
    fn model_role_display() {
        assert_eq!(format!("{}", ModelRole::Encoder), "Encoder");
        assert_eq!(format!("{}", ModelRole::Decoder), "Decoder");
        assert_eq!(format!("{}", ModelRole::Utility), "Utility");
        assert_eq!(format!("{}", ModelRole::Formatter), "Formatter");
    }

    // -----------------------------------------------------------------------
    // from_env: env var parsing
    // -----------------------------------------------------------------------

    #[test]
    fn from_env_reads_cuda_device_id() {
        // Save/restore env to avoid cross-test interference
        let _guard = EnvGuard::new(&[
            ("CUDA_DEVICE_ID", Some("2")),
            ("USE_GPU", None),
            ("VRAM_BUDGET_GB", None),
            ("TRT_CACHE_PATH", None),
            ("GPU_OPTIMIZATION", None),
            ("INTRA_THREADS", None),
            ("INTER_THREADS", None),
        ]);
        let cfg = ModelConfig::from_env();
        assert_eq!(cfg.device_id, Some(2));
    }

    #[test]
    fn from_env_reads_vram_budget() {
        let _guard = EnvGuard::new(&[
            ("VRAM_BUDGET_GB", Some("16")),
            ("USE_GPU", None),
            ("CUDA_DEVICE_ID", None),
            ("TRT_CACHE_PATH", None),
            ("GPU_OPTIMIZATION", None),
            ("INTRA_THREADS", None),
            ("INTER_THREADS", None),
        ]);
        let cfg = ModelConfig::from_env();
        assert_eq!(cfg.gpu_mem_limit_bytes, Some(16 * GB));
    }

    #[test]
    fn from_env_reads_trt_cache_path() {
        let _guard = EnvGuard::new(&[
            ("TRT_CACHE_PATH", Some("/tmp/trt")),
            ("USE_GPU", None),
            ("CUDA_DEVICE_ID", None),
            ("VRAM_BUDGET_GB", None),
            ("GPU_OPTIMIZATION", None),
            ("INTRA_THREADS", None),
            ("INTER_THREADS", None),
        ]);
        let cfg = ModelConfig::from_env();
        assert_eq!(cfg.trt_engine_cache_path.as_deref(), Some("/tmp/trt"));
    }

    #[test]
    fn from_env_reads_gpu_optimization_level() {
        let _guard = EnvGuard::new(&[
            ("GPU_OPTIMIZATION", Some("1")),
            ("USE_GPU", None),
            ("CUDA_DEVICE_ID", None),
            ("VRAM_BUDGET_GB", None),
            ("TRT_CACHE_PATH", None),
            ("INTRA_THREADS", None),
            ("INTER_THREADS", None),
        ]);
        let cfg = ModelConfig::from_env();
        assert_eq!(
            cfg.gpu_optimization_level,
            Some(GpuOptimizationLevel::Conservative)
        );
    }

    #[test]
    fn from_env_defaults_when_no_gpu_vars() {
        let _guard = EnvGuard::new(&[
            ("USE_GPU", None),
            ("CUDA_DEVICE_ID", None),
            ("VRAM_BUDGET_GB", None),
            ("TRT_CACHE_PATH", None),
            ("GPU_OPTIMIZATION", None),
            ("INTRA_THREADS", None),
            ("INTER_THREADS", None),
        ]);
        let cfg = ModelConfig::from_env();
        assert!(cfg.device_id.is_none());
        assert!(cfg.gpu_mem_limit_bytes.is_none());
        assert!(cfg.trt_engine_cache_path.is_none());
        assert!(cfg.gpu_optimization_level.is_none());
    }

    // -----------------------------------------------------------------------
    // Default VRAM budget constant
    // -----------------------------------------------------------------------

    #[test]
    fn default_vram_budget_is_20gb() {
        let expected = (20.0 * GB as f64) as u64;
        let base = ModelConfig {
            gpu_optimization_level: Some(GpuOptimizationLevel::Aggressive),
            ..ModelConfig::default()
        };
        let cfg = ModelConfig::for_role(ModelRole::Encoder, &base);
        assert_eq!(cfg.gpu_mem_limit_bytes, Some(expected));
    }

    // -----------------------------------------------------------------------
    // Helper: RAII guard for environment variable manipulation in tests
    // -----------------------------------------------------------------------

    struct EnvGuard {
        restore: Vec<(String, Option<String>)>,
    }

    impl EnvGuard {
        fn new(vars: &[(&str, Option<&str>)]) -> Self {
            let mut restore = Vec::new();
            for &(key, val) in vars {
                let prev = std::env::var(key).ok();
                restore.push((key.to_string(), prev));
                match val {
                    Some(v) => std::env::set_var(key, v),
                    None => std::env::remove_var(key),
                }
            }
            Self { restore }
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            for (key, val) in &self.restore {
                match val {
                    Some(v) => std::env::set_var(key, v),
                    None => std::env::remove_var(key),
                }
            }
        }
    }
}
