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

#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub execution_provider: ExecutionProvider,
    pub intra_threads: usize,
    pub inter_threads: usize,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            execution_provider: ExecutionProvider::default(),
            intra_threads: 4,
            inter_threads: 1,
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

        eprintln!(
            "[Config] Execution provider: {}, intra_threads: {}, inter_threads: {}",
            provider.name(),
            intra_threads,
            inter_threads
        );

        Self {
            execution_provider: provider,
            intra_threads,
            inter_threads,
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
                ort::execution_providers::CUDAExecutionProvider::default()
                    .build()
                    .error_on_failure(),
                CPUExecutionProvider::default().build(),
            ])?,

            #[cfg(feature = "tensorrt")]
            ExecutionProvider::TensorRT => builder.with_execution_providers([
                ort::execution_providers::TensorRTExecutionProvider::default().build(),
                CPUExecutionProvider::default().build().error_on_failure(),
            ])?,

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
