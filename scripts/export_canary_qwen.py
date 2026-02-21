#!/usr/bin/env python3
"""
Export NVIDIA Canary-Qwen 2.5B SALM model components to ONNX.

This script exports the AudioPerceptionModule (FastConformer encoder + modality adapter
+ linear projection) from the NeMo SALM model, then combines it with the pre-exported
Qwen3 decoder from huggingface.co/onnx-community/canary-qwen-2.5b-ONNX.

Requirements:
    pip install nemo_toolkit[speechlm2]>=2.5.0 torch onnx onnxruntime numpy

Usage:
    python scripts/export_canary_qwen.py --output-dir ./canary-qwen --skip-decoder
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class PerceptionWithoutPreprocessor(nn.Module):
    """Wraps AudioPerceptionModule but skips the preprocessor.

    Takes mel spectrogram features (already preprocessed) and runs:
    encoder -> modality_adapter -> proj -> output [B, T', D_llm]
    """

    def __init__(self, perception):
        super().__init__()
        self.encoder = perception.encoder
        self.modality_adapter = perception.modality_adapter
        self.proj = perception.proj
        self.spec_augmentation = None  # Never augment during export

    def forward(self, audio_signal, length):
        encoded, encoded_len = self.encoder(audio_signal=audio_signal, length=length)
        encoded, encoded_len = self.modality_adapter(audio_signal=encoded, length=encoded_len)
        # b, c, t -> b, t, c
        encoded = self.proj(encoded.transpose(1, 2))
        return encoded, encoded_len


class IdentityProjection(nn.Module):
    """Identity projection (pass-through) for compatibility with Rust code."""

    def forward(self, encoder_output):
        return encoder_output


def export_encoder(model, output_path: Path, opset_version: int = 17):
    """Export the full perception pipeline (encoder + adapter + proj) to ONNX.

    This exports the entire AudioPerceptionModule minus the preprocessor.
    Input: mel spectrogram features [B, n_mels, T] + length [B]
    Output: LLM-ready embeddings [B, T', D_llm] + length [B]
    """
    print("[1/5] Exporting perception pipeline as encoder.onnx...")

    # Wrap perception without preprocessor
    perception_wrapper = PerceptionWithoutPreprocessor(model.perception)
    perception_wrapper.eval()

    # Print architecture info
    print(f"  Encoder type: {type(model.perception.encoder).__name__}")
    print(f"  Modality adapter type: {type(model.perception.modality_adapter).__name__}")
    print(f"  Projection type: {type(model.perception.proj).__name__}")

    # Determine n_mels from preprocessor config
    if hasattr(model.perception, 'preprocessor') and hasattr(model.perception.preprocessor, 'featurizer'):
        n_mels = model.perception.preprocessor.featurizer.nfilt
        print(f"  n_mels: {n_mels}")
    else:
        n_mels = 128
        print(f"  n_mels: {n_mels} (default)")

    # Dummy inputs: mel features [batch, n_mels, time_frames]
    batch_size = 1
    time_frames = 1600  # ~10 seconds at 160 hop_length

    dummy_audio = torch.randn(batch_size, n_mels, time_frames)
    dummy_length = torch.tensor([time_frames], dtype=torch.long)

    # Test forward pass first
    with torch.no_grad():
        test_out, test_len = perception_wrapper(dummy_audio, dummy_length)
        print(f"  Test forward: input ({batch_size}, {n_mels}, {time_frames}) -> output {tuple(test_out.shape)}")
        print(f"  Output dim (D_llm): {test_out.shape[-1]}")

    with torch.no_grad():
        torch.onnx.export(
            perception_wrapper,
            (dummy_audio, dummy_length),
            str(output_path / "encoder.onnx"),
            input_names=["audio_signal", "length"],
            output_names=["encoder_output", "encoder_length"],
            dynamic_axes={
                "audio_signal": {0: "batch", 2: "time"},
                "length": {0: "batch"},
                "encoder_output": {0: "batch", 1: "time_out"},
                "encoder_length": {0: "batch"},
            },
            opset_version=opset_version,
            do_constant_folding=True,
        )

    size_mb = (output_path / "encoder.onnx").stat().st_size / 1e6
    print(f"  [DONE] Saved encoder.onnx ({size_mb:.1f} MB)")

    # Check for external data
    data_file = output_path / "encoder.onnx_data"
    if data_file.exists():
        data_mb = data_file.stat().st_size / 1e6
        print(f"  [DONE] External data: encoder.onnx_data ({data_mb:.1f} MB)")


def export_projection(output_path: Path, d_llm: int, opset_version: int = 17):
    """Export an identity projection for compatibility with the Rust pipeline.

    Since encoder.onnx already includes the full perception pipeline (encoder +
    modality_adapter + proj), this projection is just a pass-through identity.
    """
    print("[2/5] Exporting identity projection.onnx...")

    identity = IdentityProjection()
    identity.eval()

    dummy_input = torch.randn(1, 100, d_llm)

    with torch.no_grad():
        torch.onnx.export(
            identity,
            dummy_input,
            str(output_path / "projection.onnx"),
            input_names=["encoder_output"],
            output_names=["projected_output"],
            dynamic_axes={
                "encoder_output": {0: "batch", 1: "seq_len"},
                "projected_output": {0: "batch", 1: "seq_len"},
            },
            opset_version=opset_version,
            do_constant_folding=True,
        )

    size_mb = (output_path / "projection.onnx").stat().st_size / 1e6
    print(f"  [DONE] Saved projection.onnx ({size_mb:.1f} MB) (identity pass-through)")


def download_decoder_and_tokenizer(output_path: Path):
    """Download the pre-exported Qwen3 decoder and tokenizer from onnx-community."""
    print("[3/5] Downloading decoder and tokenizer from onnx-community...")

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("  [ERROR] huggingface_hub not installed. Run: pip install huggingface_hub")
        sys.exit(1)

    repo_id = "onnx-community/canary-qwen-2.5b-ONNX"
    local_dir = snapshot_download(
        repo_id,
        local_dir=str(output_path / "_onnx_community_download"),
        allow_patterns=[
            "*.onnx",
            "*.onnx_data",
            "*.json",
            "*.txt",
            "tokenizer.*",
            "vocab.*",
            "merges.*",
            "special_tokens_map.*",
            "added_tokens.*",
        ],
    )

    local_dir = Path(local_dir)

    import glob
    import shutil

    files_to_copy = [
        "tokenizer.json",
        "config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "vocab.json",
        "merges.txt",
        "tokenizer_config.json",
    ]

    for fname in files_to_copy:
        src = local_dir / fname
        if src.exists():
            shutil.copy2(src, output_path / fname)
            print(f"  [COPY] {fname}")

    onnx_dir = local_dir / "onnx"
    if not onnx_dir.exists():
        onnx_dir = local_dir

    for pattern in ["decoder_model_merged*.onnx*", "embed_tokens*.onnx*"]:
        for src_file in glob.glob(str(onnx_dir / pattern)):
            src_path = Path(src_file)
            shutil.copy2(src_path, output_path / src_path.name)
            print(f"  [COPY] {src_path.name} ({src_path.stat().st_size / 1e6:.1f} MB)")

    shutil.rmtree(local_dir, ignore_errors=True)
    print("  [DONE] Decoder and tokenizer downloaded")


def validate_pipeline(output_path: Path):
    """Validate the exported ONNX pipeline end-to-end."""
    print("[4/5] Validating ONNX pipeline...")

    import onnxruntime as ort

    required_files = ["encoder.onnx", "projection.onnx", "tokenizer.json", "config.json"]

    has_decoder = any(
        (output_path / f).exists()
        for f in [
            "decoder_model_merged_q4.onnx",
            "decoder_model_merged_fp16.onnx",
            "decoder_model_merged.onnx",
        ]
    )

    for f in required_files:
        if not (output_path / f).exists():
            print(f"  [WARN] Missing required file: {f}")

    if not has_decoder:
        print("  [WARN] No decoder ONNX file found")

    # Test encoder
    enc_out = None
    try:
        encoder_sess = ort.InferenceSession(str(output_path / "encoder.onnx"))
        # Print input/output info
        for inp in encoder_sess.get_inputs():
            print(f"  Encoder input: {inp.name} shape={inp.shape} dtype={inp.type}")
        for out in encoder_sess.get_outputs():
            print(f"  Encoder output: {out.name} shape={out.shape} dtype={out.type}")

        dummy_audio = np.random.randn(1, 128, 800).astype(np.float32)
        dummy_length = np.array([800], dtype=np.int64)

        enc_out = encoder_sess.run(
            None,
            {
                "audio_signal": dummy_audio,
                "length": dummy_length,
            },
        )
        print(f"  [OK] Encoder: input (1, 128, 800) -> output {enc_out[0].shape}")
    except Exception as e:
        print(f"  [FAIL] Encoder validation: {e}")

    # Test projection
    try:
        proj_sess = ort.InferenceSession(str(output_path / "projection.onnx"))
        d_llm = enc_out[0].shape[-1] if enc_out is not None else 2048
        dummy_enc = np.random.randn(1, 50, d_llm).astype(np.float32)

        proj_out = proj_sess.run(None, {"encoder_output": dummy_enc})
        print(f"  [OK] Projection: input (1, 50, {d_llm}) -> output {proj_out[0].shape}")
        # Verify identity
        if np.allclose(dummy_enc, proj_out[0]):
            print("  [OK] Projection is identity (pass-through) as expected")
    except Exception as e:
        print(f"  [FAIL] Projection validation: {e}")

    # Test embed_tokens
    try:
        embed_sess = ort.InferenceSession(str(output_path / "embed_tokens.onnx"))
        for inp in embed_sess.get_inputs():
            print(f"  Embed input: {inp.name} shape={inp.shape} dtype={inp.type}")
        for out in embed_sess.get_outputs():
            print(f"  Embed output: {out.name} shape={out.shape} dtype={out.type}")

        dummy_ids = np.array([[151644, 151645]], dtype=np.int64)
        embed_out = embed_sess.run(None, {"input_ids": dummy_ids})
        print(f"  [OK] Embed tokens: input {dummy_ids.shape} -> output {embed_out[0].shape}")
    except Exception as e:
        print(f"  [FAIL] Embed tokens validation: {e}")

    print("  [DONE] Validation complete")


def create_model_info(output_path: Path):
    """Create a model info file summarizing the export."""
    print("[5/5] Creating model info...")

    import json

    info = {
        "model_name": "canary-qwen-2.5b",
        "source": "nvidia/canary-qwen-2.5b",
        "architecture": "SALM (FastConformer + Qwen3-1.7B)",
        "language": "en",
        "sample_rate": 16000,
        "n_mels": 128,
        "max_audio_secs": 40.0,
        "notes": {
            "encoder.onnx": "Full perception pipeline: FastConformer encoder + modality_adapter + projection. Takes mel features, outputs LLM-ready embeddings.",
            "projection.onnx": "Identity pass-through (projection is included in encoder.onnx).",
        },
        "files": {},
    }

    for f in sorted(output_path.iterdir()):
        if f.is_file():
            info["files"][f.name] = f.stat().st_size

    with open(output_path / "model_info.json", "w") as fp:
        json.dump(info, fp, indent=2)

    print(f"  [DONE] Model exported to {output_path}")
    print(f"  Files: {len(info['files'])}")
    for name, size in sorted(info["files"].items()):
        print(f"    {name}: {size / 1e6:.1f} MB")


def main():
    parser = argparse.ArgumentParser(description="Export Canary-Qwen 2.5B to ONNX")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./canary-qwen",
        help="Output directory for ONNX files",
    )
    parser.add_argument(
        "--skip-encoder",
        action="store_true",
        help="Skip encoder/projection export (use existing files)",
    )
    parser.add_argument(
        "--skip-decoder",
        action="store_true",
        help="Skip decoder download (use existing files)",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version (default: 17)",
    )
    args = parser.parse_args()

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Canary-Qwen 2.5B ONNX Export")
    print("=" * 60)
    print(f"  Output: {output_path.resolve()}")
    print()

    d_llm = 2048  # Default Qwen3-1.7B hidden size

    if not args.skip_encoder:
        try:
            from nemo.collections.speechlm2.models.salm import SALM
        except ImportError:
            print("[ERROR] NeMo not installed. Install with:")
            print("  pip install nemo_toolkit[speechlm2]>=2.5.0")
            print()
            print("Alternatively, use --skip-encoder if you already have encoder.onnx and projection.onnx")
            sys.exit(1)

        print("Loading NeMo SALM model (this may take a while)...")
        model = SALM.from_pretrained("nvidia/canary-qwen-2.5b")
        model.eval()

        # Get actual D_llm from the model
        if hasattr(model, 'llm') and hasattr(model.llm, 'config'):
            d_llm = model.llm.config.hidden_size
            print(f"  D_llm (from model): {d_llm}")

        export_encoder(model, output_path, args.opset)
        export_projection(output_path, d_llm, args.opset)

        # Free memory
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        print("[SKIP] Encoder/projection export (using existing files)")
        export_projection(output_path, d_llm, args.opset)

    if not args.skip_decoder:
        download_decoder_and_tokenizer(output_path)
    else:
        print("[SKIP] Decoder download (using existing files)")

    validate_pipeline(output_path)
    create_model_info(output_path)

    print()
    print("=" * 60)
    print("  Export Complete!")
    print("=" * 60)
    print()
    print("Next steps:")
    print(f"  1. Set CANARY_QWEN_MODEL_PATH={output_path} in .env")
    print("  2. Rebuild: cargo build --release --features 'server,sortformer'")
    print("  3. Run: ./start-server.sh")
    print()


if __name__ == "__main__":
    main()
