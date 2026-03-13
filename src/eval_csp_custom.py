"""
eval_csp_custom.py — Evaluate the trained CSP-ADiT model on a custom dataset.

Given a CSV with columns:
    - 'atom_types': space-separated atomic numbers, e.g. "26 26 8 8 8"
      OR 'formula': chemical formula string, e.g. "Fe2O3"
    - 'cif': CIF string of the ground truth structure

The script:
    1. Loads the trained diffusion model checkpoint
    2. For each entry, uses atom_types as the CSP condition
    3. Generates N structures per composition (default 1)
    4. Compares each generated structure to the ground truth
    5. Reports match_rate and rms_dist, and optionally logs a WandB table

Usage:
    python src/eval_csp_custom.py \
        ckpt_path=/path/to/ldm.ckpt \
        csv_path=/path/to/your_dataset.csv \
        num_candidates=1 \
        batch_size=64 \
        device=cuda \
        visualize=true \
        save_dir=./csp_eval_results \
        logger=wandb

"""
import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

import argparse
import os
from typing import List, Dict

import numpy as np
import pandas as pd
import torch
from pymatgen.core.structure import Structure
from tqdm import tqdm

# ── path setup ────────────────────────────────────────────────────────────────
import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.eval.crystal import Crystal
from src.eval.crystal_reconstruction import CrystalReconstructionEvaluator
from src.models.ldm_module import CrystalCSPLDMLitModule
from src.utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


# ── helpers ───────────────────────────────────────────────────────────────────

def parse_atom_types(row: pd.Series) -> torch.Tensor:
    """Parse atom types from a CSV row.

    Supports two formats:
        - 'atom_types' column: space-separated atomic numbers "26 26 8 8 8"
        - 'formula' column: chemical formula "Fe2O3" parsed via pymatgen
    """
    if "atom_types" in row and pd.notna(row["atom_types"]):
        numbers = [int(x) for x in str(row["atom_types"]).split()]
        return torch.tensor(numbers, dtype=torch.long)

    if "formula" in row and pd.notna(row["formula"]):
        from pymatgen.core.composition import Composition
        from src.eval.crystal import chemical_symbols
        comp = Composition(row["formula"])
        atom_list = []
        for el, count in comp.items():
            symbol = str(el)
            atomic_num = chemical_symbols.index(symbol)
            atom_list.extend([atomic_num] * int(count))
        return torch.tensor(atom_list, dtype=torch.long)

    raise ValueError(f"Row has neither 'atom_types' nor 'formula' column: {row}")


def parse_gt_structure(row: pd.Series) -> Structure:
    """Parse ground truth structure from CIF string in CSV row."""
    if "cif" in row and pd.notna(row["cif"]):
        return Structure.from_str(row["cif"], fmt="cif")
    raise ValueError(f"Row has no 'cif' column: {row}")


def structure_to_arrays(structure: Structure, sample_idx: int) -> Dict:
    """Convert a pymatgen Structure to the array dict format used by Crystal."""
    lattice = structure.lattice
    return {
        "atom_types": np.array([site.specie.number for site in structure], dtype=np.int64),
        "frac_coords": structure.frac_coords.astype(np.float32),
        "lengths": np.array(lattice.abc, dtype=np.float32),
        "angles": np.array(lattice.angles, dtype=np.float32),
        "sample_idx": sample_idx,
    }


# ── main evaluation ───────────────────────────────────────────────────────────

def evaluate_csp(
    ckpt_path: str,
    csv_path: str,
    num_candidates: int = 1,
    batch_size: int = 64,
    device: str = "cuda",
    visualize: bool = False,
    save_dir: str = "./csp_eval_results",
    use_wandb: bool = False,
    wandb_name: str = "eval_csp_custom",
):
    os.makedirs(save_dir, exist_ok=True)

    # ── Load model ────────────────────────────────────────────────────────────
    log.info(f"Loading checkpoint: {ckpt_path}")
    model = CrystalCSPLDMLitModule.load_from_checkpoint(
        ckpt_path, map_location=device
    )
    model.eval()
    model = model.to(device)
    log.info("Model loaded successfully.")

    # ── Load CSV ──────────────────────────────────────────────────────────────
    log.info(f"Loading dataset: {csv_path}")
    df = pd.read_csv(csv_path)
    log.info(f"Dataset size: {len(df)} entries")

    # ── Parse atom types and ground truth structures ───────────────────────────
    all_atom_types = []
    all_gt_structures = []
    valid_indices = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Parsing dataset"):
        try:
            atom_types = parse_atom_types(row)
            gt_structure = parse_gt_structure(row)
            all_atom_types.append(atom_types)
            all_gt_structures.append(gt_structure)
            valid_indices.append(idx)
        except Exception as e:
            log.warning(f"Skipping row {idx}: {e}")

    log.info(f"Successfully parsed {len(all_atom_types)} entries.")

    # ── Initialize evaluator ──────────────────────────────────────────────────
    evaluator = CrystalReconstructionEvaluator()

    # ── Generate structures ───────────────────────────────────────────────────
    log.info(f"Generating {num_candidates} candidate(s) per composition...")

    with torch.no_grad():
        all_candidate_metrics = []  # collect per-candidate metrics for final summary
    for cand_idx in range(num_candidates):
            log.info(f"Generating candidate {cand_idx + 1}/{num_candidates}")
            evaluator.clear()

            for batch_start in tqdm(
                range(0, len(all_atom_types), batch_size),
                desc=f"  Candidate {cand_idx + 1}",
            ):
                batch_atom_types = all_atom_types[batch_start: batch_start + batch_size]
                actual_bs = len(batch_atom_types)

                # Generate structures conditioned on atom types
                out, batch_info = model.sample_and_decode(
                    num_nodes_bincount=model.num_nodes_bincount,
                    batch_size=actual_bs,
                    cfg_scale=model.hparams.sampling.cfg_scale,
                    atom_types_per_sample=batch_atom_types,
                )

                # Collect predictions and ground truths
                start_idx = 0
                for i, num_atom in enumerate(batch_info["num_atoms"].tolist()):
                    global_idx = batch_start + i

                    # Predicted structure arrays
                    _atom_types = batch_info["atom_types"][start_idx: start_idx + num_atom]
                    _frac_coords = out["frac_coords"].narrow(0, start_idx, num_atom)
                    _lengths = out["lengths"][i] * float(num_atom) ** (1 / 3)
                    _angles = torch.rad2deg(out["angles"][i])

                    evaluator.append_pred_array({
                        "atom_types": _atom_types.cpu().numpy(),
                        "frac_coords": _frac_coords.cpu().numpy(),
                        "lengths": _lengths.cpu().numpy(),
                        "angles": _angles.cpu().numpy(),
                        "sample_idx": global_idx,
                    })

                    # Ground truth structure arrays
                    evaluator.append_gt_array(
                        structure_to_arrays(all_gt_structures[global_idx], global_idx)
                    )

                    start_idx += num_atom

            # ── Compute metrics ───────────────────────────────────────────────
            metrics = evaluator.get_metrics(
                save=visualize,
                save_dir=os.path.join(save_dir, f"candidate_{cand_idx}"),
            )

            all_candidate_metrics.append(metrics)

            match_rate   = metrics["match_rate"].float().mean().item()
            rms_dist     = metrics["rms_dist"].mean().item() if len(metrics["rms_dist"]) > 0 else float("inf")

            # Compute comp_match_rate and struct_valid_rate directly from crystal lists
            # (CrystalReconstructionEvaluator stores pred/gt crystal lists after get_metrics)
            comp_match_rate   = float("nan")
            struct_valid_rate = float("nan")
            if len(evaluator.pred_crys_list) > 0 and len(evaluator.gt_crys_list) > 0:
                comp_matches = [
                    sorted(p.atom_types) == sorted(g.atom_types)
                    for p, g in zip(evaluator.pred_crys_list, evaluator.gt_crys_list)
                ]
                struct_valids = [c.struct_valid for c in evaluator.pred_crys_list]
                comp_match_rate   = sum(comp_matches) / len(comp_matches)
                struct_valid_rate = sum(struct_valids) / len(struct_valids)

            log.info(
                f"Candidate {cand_idx + 1} metrics:\n"
                f"  match_rate       : {match_rate:.4f}  (StructureMatcher RMSD < threshold)\n"
                f"  rms_dist         : {rms_dist:.4f} A\n"
                f"  comp_match_rate  : {comp_match_rate:.4f}  (pred atom types == gt atom types)\n"
                f"  struct_valid_rate: {struct_valid_rate:.4f}  (min pairwise dist > 0.5 A, volume > 0.1)"
            )

            # ── WandB logging ─────────────────────────────────────────────────
            if use_wandb:
                import wandb
                from lightning.pytorch.loggers import WandbLogger
                wandb.log({
                    f"csp/match_rate_cand{cand_idx}":    match_rate,
                    f"csp/rms_dist_cand{cand_idx}":      rms_dist,
                    f"csp/comp_match_rate_cand{cand_idx}":   comp_match_rate,
                    f"csp/struct_valid_rate_cand{cand_idx}": struct_valid_rate,
                })
                if visualize:
                    table = evaluator.get_wandb_table(
                        current_epoch=cand_idx,
                        save_dir=os.path.join(save_dir, f"candidate_{cand_idx}"),
                    )
                    wandb.log({f"csp_samples_candidate_{cand_idx}": table})

    # ── Overall summary across all candidates ────────────────────────────────
    log.info("\n" + "="*60)
    log.info("OVERALL EVALUATION SUMMARY")
    log.info("="*60)
    log.info(f"  Dataset size     : {len(all_candidate_metrics[0]['match_rate'])} structures")
    log.info(f"  Num candidates   : {num_candidates}")

    # Per-candidate match rates
    candidate_match_rates = [m["match_rate"].float().mean().item() for m in all_candidate_metrics]
    for i, mr in enumerate(candidate_match_rates):
        log.info(f"  Candidate {i+1} match_rate: {mr:.4f}")

    # Best-of-N: a structure is matched if ANY candidate matches it
    if num_candidates > 1:
        per_sample_any_match = torch.stack(
            [m["match_rate"] for m in all_candidate_metrics], dim=0
        ).any(dim=0).float()
        best_of_n_match_rate = per_sample_any_match.mean().item()
        log.info(f"  Best-of-{num_candidates} match_rate: {best_of_n_match_rate:.4f}  (any candidate matches gt)")

    log.info(f"  Overall match_rate (mean across candidates): {sum(candidate_match_rates)/len(candidate_match_rates):.4f}")
    log.info("="*60)

    log.info("Evaluation complete.")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CSP evaluation on custom dataset")
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--num_candidates", type=int, default=1,
                        help="Number of structure candidates to generate per composition")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--visualize", action="store_true",
                        help="Save CIF files and log WandB table")
    parser.add_argument("--save_dir", type=str, default="./csp_eval_results")
    parser.add_argument("--wandb", action="store_true", dest="use_wandb",
                        help="Log results to WandB")
    parser.add_argument("--wandb_name", type=str, default="eval_csp_custom")
    args = parser.parse_args()

    if args.use_wandb:
        import wandb
        wandb.init(project="csp-adit", name=args.wandb_name)

    evaluate_csp(
        ckpt_path=args.ckpt_path,
        csv_path=args.csv_path,
        num_candidates=args.num_candidates,
        batch_size=args.batch_size,
        device=args.device,
        visualize=args.visualize,
        save_dir=args.save_dir,
        use_wandb=args.use_wandb,
        wandb_name=args.wandb_name,
    )