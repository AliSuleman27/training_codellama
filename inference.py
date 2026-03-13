#!/usr/bin/env python3
"""
Inference script for the trained CodeLLaMA-7B-Python LoRA adapter.
Use this to generate predictions with your trained adapter.

Usage:
  # On RunPod (8bit):
  python inference.py --quantize 8bit

  # On local 6GB GPU (4bit):
  python inference.py --adapter_path ./adapter --quantize 4bit

  # Single prompt:
  python inference.py --quantize 8bit --prompt "def add(a, b):\\n<FILL_ME>"
"""

import torch
import argparse
from pathlib import Path
from typing import Optional

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel
except ImportError:
    print("Error: transformers or peft not installed. Install with: pip install transformers peft")
    import sys
    sys.exit(1)


class AdapterInference:
    """Inference class for CodeLLaMA adapter."""

    def __init__(
        self,
        adapter_path: str,
        base_model_name: str = "codellama/CodeLlama-7b-Python-hf",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        dtype: torch.dtype = torch.float16,
        quantize: Optional[str] = None,
    ):
        """Initialize the adapter for inference.

        Args:
            quantize: None for FP16, "4bit" for 4-bit quantization (~4GB VRAM),
                      "8bit" for 8-bit quantization (~7GB VRAM).
        """
        self.device = device
        self.dtype = dtype

        print("=" * 80)
        print("LOADING MODEL")
        print("=" * 80)

        print(f"\n Base Model: {base_model_name}")
        print(f" Adapter: {adapter_path}")
        print(f" Device: {device}")
        print(f" Data Type: {dtype}")
        if quantize:
            print(f" Quantization: {quantize}")

        # Configure quantization
        quantization_config = None
        if quantize == "4bit":
            print("\n Loading with 4-bit quantization (low VRAM mode)...")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )
        elif quantize == "8bit":
            print("\n Loading with 8-bit quantization...")
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
            )

        # Load base model
        print("\n Loading base model...")
        load_kwargs = dict(
            trust_remote_code=True,
            device_map="auto" if quantize else device,
        )
        if quantization_config:
            load_kwargs["quantization_config"] = quantization_config
        else:
            load_kwargs["torch_dtype"] = dtype

        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            **load_kwargs,
        )

        # Load adapter
        print(" Loading LoRA adapter...")
        self.model = PeftModel.from_pretrained(
            self.model,
            adapter_path,
        )

        # Load tokenizer
        print(" Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            trust_remote_code=True,
        )
        self.tokenizer.padding_side = "left"

        # Only move to device if not using quantization (quantized models use device_map="auto")
        if not quantize:
            self.model.to(device)
        self.model.eval()

        print(" Model loaded successfully!")
        print("=" * 80)

    def generate(
        self,
        prompt: str,
        max_length: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        num_return_sequences: int = 1,
        do_sample: bool = True,
    ) -> list:
        """Generate predictions."""

        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        ).to(self.model.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                num_return_sequences=num_return_sequences,
                do_sample=do_sample,
                early_stopping=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode
        predictions = []
        for output in outputs:
            decoded = self.tokenizer.decode(output, skip_special_tokens=True)
            predictions.append(decoded)

        return predictions

    def generate_bugfix(
        self,
        buggy_code: str,
        max_length: int = 256,
        num_variations: int = 1,
    ) -> dict:
        """Generate bug fixes for given buggy code."""

        generations = self.generate(
            prompt=buggy_code,
            max_length=max_length,
            num_return_sequences=num_variations,
            do_sample=True,
            temperature=0.7,
        )

        return {
            "input": buggy_code,
            "generations": generations,
            "num_variations": num_variations,
        }


def interactive_mode(inference):
    """Interactive prompt for generating fixes."""
    print("\n" + "=" * 80)
    print("INTERACTIVE MODE")
    print("=" * 80)
    print("Enter buggy code snippets. Type 'exit' to quit.\n")

    while True:
        try:
            print("\n" + "-" * 80)
            print("Enter buggy code (end with empty line):")
            lines = []
            while True:
                line = input()
                if not line:
                    break
                lines.append(line)

            if not lines:
                continue

            buggy_code = "\n".join(lines)

            if buggy_code.lower() == "exit":
                break

            print("\nGenerating fix...")
            result = inference.generate_bugfix(buggy_code, num_variations=1)

            print("\nInput:")
            print(result["input"][:200] + ("..." if len(result["input"]) > 200 else ""))

            print("\nGenerated Fix:")
            for i, gen in enumerate(result["generations"], 1):
                print(f"\nVariation {i}:")
                print(gen[:500] + ("..." if len(gen) > 500 else ""))

        except KeyboardInterrupt:
            print("\n\nInterrupted.")
            break
        except Exception as e:
            print(f"Error: {e}")
            continue


def main():
    parser = argparse.ArgumentParser(
        description="Inference for CodeLLaMA-7B-Python LoRA adapter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Interactive mode (RunPod):
    python inference.py --quantize 8bit

  Single prompt:
    python inference.py --quantize 8bit --prompt "def buggy(): <FILL_ME>"

  Local 6GB GPU:
    python inference.py --adapter_path ./adapter --quantize 4bit
        """
    )

    parser.add_argument(
        "--adapter_path",
        type=str,
        default="/workspace/output/codellama-7b-python-adapter",
        help="Path to trained LoRA adapter"
    )
    parser.add_argument(
        "--base_model_name",
        type=str,
        default="codellama/CodeLlama-7b-Python-hf",
        help="Base model name or path"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Single prompt to generate for"
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default=None,
        help="File containing buggy code"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Save outputs to file"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=256,
        help="Maximum generation length"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Nucleus sampling p"
    )
    parser.add_argument(
        "--num_variations",
        type=int,
        default=1,
        help="Number of variations to generate"
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda or cpu)"
    )
    parser.add_argument(
        "--quantize",
        type=str,
        choices=["4bit", "8bit"],
        default=None,
        help="Quantization mode: '4bit' (~4GB VRAM, for 6GB GPUs) or '8bit' (~7GB VRAM). Default: None (FP16, needs ~13GB)"
    )

    args = parser.parse_args()

    # Initialize inference
    inference = AdapterInference(
        adapter_path=args.adapter_path,
        base_model_name=args.base_model_name,
        device=args.device,
        quantize=args.quantize,
    )

    # Generate from prompt
    if args.prompt:
        print("\nGenerating...")
        result = inference.generate_bugfix(args.prompt, max_length=args.max_length)

        print("\nInput:")
        print(result["input"])

        print("\nGenerated Fixes:")
        for i, gen in enumerate(result["generations"], 1):
            print(f"\nVariation {i}:")
            print(gen)

        if args.output_file:
            with open(args.output_file, 'w') as f:
                f.write("Input:\n" + result["input"] + "\n\n")
                for i, gen in enumerate(result["generations"], 1):
                    f.write(f"Variation {i}:\n{gen}\n\n")
            print(f"\nResults saved to {args.output_file}")

    # Generate from file
    elif args.input_file:
        with open(args.input_file, 'r') as f:
            buggy_code = f.read()

        print("\nGenerating...")
        result = inference.generate_bugfix(
            buggy_code,
            max_length=args.max_length,
            num_variations=args.num_variations
        )

        print("\nInput (from file):")
        print(result["input"][:300] + ("..." if len(result["input"]) > 300 else ""))

        print("\nGenerated Fixes:")
        for i, gen in enumerate(result["generations"], 1):
            print(f"\nVariation {i}:")
            print(gen[:500] + ("..." if len(gen) > 500 else ""))

        if args.output_file:
            with open(args.output_file, 'w') as f:
                f.write("Input:\n" + result["input"] + "\n\n")
                for i, gen in enumerate(result["generations"], 1):
                    f.write(f"Variation {i}:\n{gen}\n\n")
            print(f"\nResults saved to {args.output_file}")

    # Interactive mode
    else:
        interactive_mode(inference)


if __name__ == "__main__":
    main()
