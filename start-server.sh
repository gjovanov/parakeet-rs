#!/bin/bash

# SRT Stream Configuration
export SRT_ENCODER_IP="10.84.17.100"
export SRT_CHANNELS='[{"name":"ORF1","port":"24001"},{"name":"ORF2","port":"24002"},{"name":"KIDS","port":"24011"},{"name":"ORFS","port":"24004"},{"name":"ORF-B","port":"24013"},{"name":"ORF-K","port":"24019"},{"name":"ORF-NOE","port":"24012"},{"name":"ORF-OOE","port":"24014"},{"name":"ORF-S","port":"24015"},{"name":"ORF-ST","port":"24018"},{"name":"ORF-T","port":"24016"},{"name":"ORF-V","port":"24017"},{"name":"ORF-W","port":"24011"},{"name":"ORF-SI","port":"24016"}]'
export SRT_LATENCY="200000"
export SRT_RCVBUF="2097152"

# Optional: Override auto-detected max parallel threads
# (auto-detection uses: min(available_ram_gb - 4) / 2.5, cpu_cores), max 8)
# export MAX_PARALLEL_THREADS=2

# Limit max concurrent sessions to prevent memory exhaustion
export MAX_CONCURRENT_SESSIONS="${MAX_CONCURRENT_SESSIONS:-3}"

# Enable GPU acceleration (CUDA)
export USE_GPU=true

# Add CUDA 12.8 libraries to path (includes cuDNN 9)
# Also add ONNX Runtime provider libraries
ORT_LIB_PATH="/home/ubuntu/parakeet-rs/target/release/examples"
export LD_LIBRARY_PATH="/usr/local/cuda-12.8/lib:/usr/local/cuda-12.8/lib64:${ORT_LIB_PATH}:${LD_LIBRARY_PATH}"

# Canary model path (for multilingual support)
export CANARY_MODEL_PATH="/home/ubuntu/parakeet-rs/canary"

exec ./target/release/examples/webrtc_transcriber \
  --port 80 \
  --tdt-model ./tdt \
  --diar-model ./diar_streaming_sortformer_4spk-v2.onnx \
  --vad-model ./silero_vad.onnx \
  --frontend ./frontend \
  --media-dir ./media \
  --public-ip "10.84.17.72" \
  --max-sessions "${MAX_CONCURRENT_SESSIONS}" \
  --speedy
