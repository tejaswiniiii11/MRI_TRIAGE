"""
CLI & Helper Tool to display slice-by-slice attention percentages
for any patient in the dataset.

Usage:
    python get_slice_attention.py --patient_id 995030-000001
    python get_slice_attention.py --patient_id 995030-000376 --age 62
"""

import os
import sys
import argparse
import pandas as pd

# Add CODE directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import Config
from src.inference import Module1InferenceEngine

def main():
    parser = argparse.ArgumentParser(description="Inspect Position-Aware Attention Percentage for Each Slice")
    parser.add_argument("--patient_id", type=str, required=True, help="Patient ID (e.g. 995030-000001)")
    parser.add_argument("--age", type=float, default=None, help="Patient age (optional; if omitted, looked up from bridge.xlsx)")
    parser.add_argument("--top_k", type=int, default=5, help="Number of top slices to highlight (default: 5)")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_module1_age_aware.pt")
    args = parser.parse_args()

    ckpt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.checkpoint)
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(Config.CHECKPOINTS_DIR, os.path.basename(args.checkpoint))

    engine = Module1InferenceEngine(ckpt_path)

    # Look up age and ground truth if available in bridge.xlsx
    age = args.age
    ground_truth = "Unknown"
    bridge_path = Config.BRIDGE_EXCEL
    if os.path.exists(bridge_path):
        df = pd.read_excel(bridge_path)
        row = df[df['ANON_PatientID'] == args.patient_id]
        if not row.empty:
            ground_truth = str(row['Classification'].values[0])
            if age is None and 'Age' in row.columns:
                age = float(row['Age'].values[0])

    if age is None and engine.is_age_aware:
        age = 50.0  # default fallback if age-aware model is used

    result = engine.get_patient_slice_attention(args.patient_id, age=age)

    print("\n" + "=" * 65)
    print(f"PATIENT SLICE ATTENTION REPORT: {args.patient_id}")
    print("=" * 65)
    print(f"Ground Truth Label:       {ground_truth}")
    print(f"Model Prediction:         {result['prediction']}")
    print(f"Abnormality Probability:  {result['abnormality_probability']:.2%}")
    if age is not None:
        print(f"Patient Age Used:         {age:.1f} years")
    print(f"Total Evaluated Slices:   {result['total_slices']}")
    print("-" * 65)

    print(f"\n{'Slice #':<10}{'Relative Pos':<16}{'Attention %':<16}{'Visual Contribution'}")
    print("-" * 65)
    for s in result['slices']:
        bar = "#" * int(s['attention_percentage'] * 2)
        print(f"{s['slice_index']:<10}{s['relative_position']:<16.3f}{s['attention_percentage']:<6.2f}%         {bar}")

    print("\n" + "-" * 65)
    print(f"TOP {min(args.top_k, result['total_slices'])} CONTRIBUTING SLICES:")
    print("-" * 65)
    for rank, s in enumerate(result['top_contributing_slices'][:args.top_k], 1):
        print(f"  Rank {rank}: Slice #{s['slice_index']} (Relative Pos: {s['relative_position']:.3f}) -> {s['attention_percentage']:.2f}% of total attention")
    print("=" * 65 + "\n")

if __name__ == "__main__":
    main()
