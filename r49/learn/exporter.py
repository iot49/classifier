
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from fastai.vision.all import *  # noqa: F403
from fastai.vision.all import CrossEntropyLossFlat, error_rate, vision_learner

try:
    import onnx
    import onnxruntime as ort
    from onnxconverter_common.float16 import convert_float_to_float16
    from onnxruntime.quantization import QuantType, quantize_dynamic
except ImportError as e:
    print(f"Warning: ONNX/ORT libraries not installed. Export will fail. Error: {e}")

from .. import R49DataLoaders, R49Dataset
from .config import LearnerConfig

KEEP_ONNX = True


class Exporter(LearnerConfig):
    
    def __init__(self, model_name: str):
        super().__init__(model_name)
        
        # Load DataLoaders (needed for validation and sample input)
        ds = R49Dataset(self.data_dir.rglob("**/*.r49"), dpt=self.dpt, size=int(1.5*self.size), labels=self.labels)
        self._dls = R49DataLoaders.from_dataset(ds, valid_pct=self.valid_pct, crop_size=self.size, bs=self.batch_size, vocab=self.labels)
        
        # Load Model
        arch = self.get_architecture(self.arch_name)
        self._learn_obj = vision_learner(self._dls, arch, metrics=error_rate, loss_func=CrossEntropyLossFlat())
        
        model_path = self.model_dir / "model.pth"
        if model_path.exists():
            print(f"Loading weights from {model_path}")
            saved_model = torch.load(model_path, map_location=self._dls.device, weights_only=False)
            if isinstance(saved_model, torch.nn.Module):
                self._learn_obj.model = saved_model
            else:
                self._learn_obj.model.load_state_dict(saved_model)
        else:
            raise FileNotFoundError(f"Model file not found at {model_path}")
            
    def export(self):
        """
        Exports the model to ONNX (FP32, FP16, Int8) and ORT formats.
        """
        print(f"Exporting model '{self._model_name}'...")
        self._learn_obj.model.eval()
        
        # Dummy input for export
        dummy_input = torch.randn(1, 3, self.size, self.size, device=self._dls.device)
        
        # Paths
        # Paths
        export_dir = self.model_dir / "export"
        if export_dir.exists():
            print(f"Propelling old export files in {export_dir}...")
            shutil.rmtree(export_dir)
        export_dir.mkdir(exist_ok=True)
        
        # Export both ONNX and ORT formats.
        # ONNX is the standard interchange format.
        # ORT format is optimized for ONNX Runtime (mobile/web) and is smaller/faster to load.
        onnx_path_fp32 = export_dir / "model_fp32.onnx"
        onnx_path_fp16 = export_dir / "model_fp16.onnx"
        onnx_path_int8 = export_dir / "model_int8.onnx"
        
        # 1. Export FP32 ONNX
        print(f"Exporting FP32 ONNX to {onnx_path_fp32}...")
        torch.onnx.export(
            self._learn_obj.model, 
            dummy_input, 
            onnx_path_fp32,
            export_params=True,
            opset_version=14, # Reverting to 14 as 17 might be too high for some backends, and legacy exporter works well with 11-14
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
            dynamo=False,
        )
        
        # 2. Convert to FP16
        print(f"Converting to FP16 ONNX at {onnx_path_fp16}...")
        model_fp32 = onnx.load(str(onnx_path_fp32))
        model_fp16 = convert_float_to_float16(model_fp32)
        onnx.save(model_fp16, str(onnx_path_fp16))
        
        # 3. Quantize to Int8
        print(f"Quantizing to Int8 ONNX at {onnx_path_int8}...")
        # Note: Dynamic quantization usually keeps first/last layers as float if they are sensitive operations
        # but pure fully connected or conv layers might be quantized. 
        # For vision models, static quantization is often better but requires calibration data.
        # Here we use dynamic for simplicity as requested.
        quantize_dynamic(
            model_input=onnx_path_fp32,
            model_output=onnx_path_int8,
            weight_type=QuantType.QUInt8
        )
        
        # 4. Convert to ORT format
        print("Converting ONNX models to ORT format...")
        for onnx_file in [onnx_path_fp32, onnx_path_fp16, onnx_path_int8]:
            if onnx_file.exists():
                self._convert_to_ort(onnx_file)
                
        # 5. Validation
        print("\n=== Validation Results ===")
        metrics = {
            "train_samples": len(self._dls.train_ds),
            "valid_samples": len(self._dls.valid_ds),
            "error_rates": {},
            "sizes_mb": {}
        }
        
        loss, err = self.validate(self._learn_obj.model, "PyTorch (FP32)")
        metrics["error_rates"]["PyTorch (FP32)"] = err
        metrics["sizes_mb"]["PyTorch (FP32)"] = (self.model_dir / "model.pth").stat().st_size / (1024 * 1024)
        
        ort_path_fp32 = onnx_path_fp32.with_suffix(".ort")
        ort_path_fp16 = onnx_path_fp16.with_suffix(".ort")
        ort_path_int8 = onnx_path_int8.with_suffix(".ort")

        err = self.validate_onnx(ort_path_fp32, "ORT (FP32)")
        metrics["error_rates"]["ORT (FP32)"] = err
        metrics["sizes_mb"]["ORT (FP32)"] = ort_path_fp32.stat().st_size / (1024 * 1024) if ort_path_fp32.exists() else 0
        
        err = self.validate_onnx(ort_path_fp16, "ORT (FP16)")
        metrics["error_rates"]["ORT (FP16)"] = err
        metrics["sizes_mb"]["ORT (FP16)"] = ort_path_fp16.stat().st_size / (1024 * 1024) if ort_path_fp16.exists() else 0
        
        err = self.validate_onnx(ort_path_int8, "ORT (Int8)")
        metrics["error_rates"]["ORT (Int8)"] = err
        metrics["sizes_mb"]["ORT (Int8)"] = ort_path_int8.stat().st_size / (1024 * 1024) if ort_path_int8.exists() else 0
        
        # 6. Copy config.json to export directory
        config_src = self.model_dir / "config.json"
        config_dst = export_dir / "model.config"
        if config_src.exists():
            print(f"Copying {config_src.name} -> {config_dst.name}")
            shutil.copy(str(config_src), str(config_dst))
            
        # 7. Write README.md with metrics
        notes_str = self.get_notes(metrics)
        readme_path = export_dir / "README.md"
        print(f"Writing {readme_path.name}")
        readme_path.write_text(notes_str)
            
        return metrics
        
    def _convert_to_ort(self, onnx_path: Path):
        """
        Converts an ONNX file to ORT format with runtime optimizations.
        Renames the output to replace the original extension with .ort and .config,
        keeping the optimized versions and removing intermediates.
        """
        # We expect the tool to generate:
        # [stem].with_runtime_opt.ort
        # [stem].required_operators.with_runtime_opt.config
        
        # We want to rename them to:
        # [stem].ort
        # [stem]_operators.config
        
        print(f"Converting {onnx_path} to ORT...")
        
        # Use subprocess to call the tool
        cmd = [
            sys.executable, "-m", "onnxruntime.tools.convert_onnx_models_to_ort", 
            str(onnx_path)
        ]
        
        try:
            # excessive output can be annoying, capturing it
            _ = subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            print(f"Failed to convert {onnx_path} to ORT: {e}")
            if e.stderr:
                print(f"Error output: {e.stderr.decode()}")
            return

        # Define expected generated filenames
        generated_ort = onnx_path.with_name(f"{onnx_path.stem}.with_runtime_opt.ort")
        generated_config = onnx_path.with_name(f"{onnx_path.stem}.required_operators.with_runtime_opt.config")
        
        # Define target filenames
        target_ort = onnx_path.with_suffix(".ort")
        target_config = onnx_path.with_name(f"{onnx_path.stem}_operators.config")
        
        # Move/Rename optimized ORT file
        if generated_ort.exists():
            print(f"  Renaming {generated_ort.name} -> {target_ort.name}")
            shutil.move(str(generated_ort), str(target_ort))
        else:
            print(f"Warning: Expected optimized file {generated_ort} not found.")

        # Move/Rename config file
        if generated_config.exists():
            print(f"  Renaming {generated_config.name} -> {target_config.name}")
            shutil.move(str(generated_config), str(target_config))
        else:
            print(f"Warning: Expected config file {generated_config} not found.")
            
        # Cleanup: Remove the unoptimized .ort file if it exists (generated by the tool by default)
        # Note: shutil.move overwrites, so unopt_ort (same path as target_ort) is effectively replaced.
        
        # Let's clean up the unoptimized config
        unopt_config = onnx_path.with_name(f"{onnx_path.stem}.required_operators.config")
        if unopt_config.exists():
            unopt_config.unlink()
        
        # Finally, remove the .onnx file after conversion as requested
        if not KEEP_ONNX and onnx_path.exists():
            print(f"  Removing {onnx_path.name}")
            onnx_path.unlink()
        
        print(f"  Finished ORT conversion for {onnx_path.stem}")

    def validate(self, model, name: str):
        # Validate PyTorch model
        # Using fastai's validate is easiest for PyTorch
        print(f"Validating {name}...")
        loss, err = self._learn_obj.validate()
        print(f"[{name}] Error Rate: {err:.4f}, Loss: {loss:.4f}")
        return loss, err

    def validate_onnx(self, model_path: Path, name: str):
        if not model_path.exists():
            print(f"[{name}] Skipped (file not found)")
            return 0.0
            
        print(f"Validating {name}...")
        session = ort.InferenceSession(str(model_path), providers=['CPUExecutionProvider'])
        input_info = session.get_inputs()[0]
        input_name = input_info.name
        input_type = input_info.type
        
        # Check if model expects float16
        is_fp16 = 'float16' in input_type
        
        correct = 0
        total = 0
        
        # Iterate validation set
        # Note: This might be slow for large sets as we are running on CPU primarily
        # To make it faster we could batch input if the session supports it (dynamic axes used above)
        for batch in self._dls.valid:
            imgs, labels = batch
            imgs_np = imgs.cpu().numpy()
            
            # Run inference
            try:
                if is_fp16:
                    imgs_np = imgs_np.astype(np.float16)
                    
                outputs = session.run(None, {input_name: imgs_np})
                preds = outputs[0]
                pred_idxs = np.argmax(preds, axis=1)  # pyright: ignore[reportCallIssue]
                
                correct += (pred_idxs == labels.cpu().numpy()).sum()
                total += len(labels)
            except Exception as e:
                print(f"Error during inference: {e}")
                break
                
        acc = correct / total if total > 0 else 0
        err = 1.0 - acc
        print(f"[{name}] Error Rate: {err:.4f}")
        return err

    def get_notes(self, metrics: dict) -> str:
        """
        Generates markdown notes for the exported model based on provided metrics.
        """
        notes = [f"# {self._model_name} Exported Model\n"]
        notes.append("| Model | Error Rate | Size (MB) |")
        notes.append("| --- | --- | --- |")
        
        # Sort model names for consistent output
        for model in sorted(metrics["error_rates"].keys()):
            err = metrics["error_rates"][model]
            size = metrics["sizes_mb"].get(model, 0)
            notes.append(f"| {model} | {err:.2%} | {size:.2f} |")
            
        notes.append("\n**Dataset Info:**")
        notes.append(f"- Training Samples: {metrics['train_samples']}")
        notes.append(f"- Validation Samples: {metrics['valid_samples']}")
        notes.append(f"- Labels: {', '.join(self.labels)}")
        
        return "\n".join(notes)
    
    def release(self, tag: str, release_candidate: bool = True):
        metrics = self.export()
        print(f"Creating GitHub release '{tag}'...")
        
        export_dir = self.model_dir / "export"
        files_to_upload = list(export_dir.glob("*.ort")) + list(export_dir.glob("*_operators.config"))
        
        if KEEP_ONNX:
            files_to_upload += list(export_dir.glob("*.onnx"))
        
        # Include model.config if it exists in export dir
        model_config_path = export_dir / "model.config"
        if model_config_path.exists():
            files_to_upload.append(model_config_path)
        
        # For backwards compatibility/safety, try to include top-level config.json as well
        config_path = self.model_dir / "config.json"
        if config_path.exists() and config_path not in files_to_upload:
            files_to_upload.append(config_path)
        
        if not files_to_upload:
            print("No files found to upload.")
            return

        # Generate Release Notes
        notes_str = self.get_notes(metrics)

        cmd = [
            "gh", "release", "create", tag,
            "--title", f"Model {self._model_name} {tag}",
            "--notes", notes_str,
        ]
        
        if release_candidate:
            cmd.append("--prerelease")
            
        cmd += [str(f) for f in files_to_upload]
        
        try:
            subprocess.run(cmd, check=True)
            print(f"Successfully created release {tag}")
        except subprocess.CalledProcessError as e:
            print(f"Failed to create release: {e}") 