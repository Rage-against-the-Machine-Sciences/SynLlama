import argparse
import re
import numpy as np
import pandas as pd
from collections import defaultdict

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

from tdc import Evaluator


def canonicalize(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # canonical form
    # drop stereochemistry (chirality and double bond geometry)
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)  # TODO: check


def tanimoto_sim(s1, s2):
    m1 = Chem.MolFromSmiles(s1)
    m2 = Chem.MolFromSmiles(s2)

    if m1 is None or m2 is None:
        return 0.0

    fp1 = AllChem.GetMorganFingerprintAsBitVect(m1, 2, 2048)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(m2, 2, 2048)

    return DataStructs.TanimotoSimilarity(fp1, fp2)


def calculate_diversity(analogue_thresh: float, evaluator: Evaluator, df: pd.DataFrame, total: int):
    rxn_pattern = re.compile(r"R\d+")
    product_good_pathways = defaultdict(list)
    bb_good_pathways = defaultdict(set)

    # success_targets = set()
    recon_targets = set()

    for _, row in df.iterrows():
        target = row["target"]
        analog = row["smiles"]
        # score = row["score"]
        # if score < args.score_threshold:
        #     continue

        sim = tanimoto_sim(target, analog)
        # analog filtering
        if sim < analogue_thresh:
            continue
        # success_targets.add(target)

        if target == analog:
            recon_targets.add(target)

        product_good_pathways[target].append(analog)

        synthesis = row.get("synthesis", None)
        if synthesis is None or pd.isna(synthesis):
            continue
        tokens = synthesis.split(";")

        for t in tokens:
            if rxn_pattern.match(t):
                continue
            smi = canonicalize(t)
            if smi:
                bb_good_pathways[target].add(smi)

    # success rate
    # print(f"Success rate: {len(success_targets)}/{total} = {len(success_targets)/total:.3f}")

    # reconstruction rate
    print(f"Reconstruction rate: {len(recon_targets)}/{total} = {len(recon_targets)/total:.3f}")

    # product diversity
    product_diversities = []

    plural_analog_count = 0
    for products in product_good_pathways.values():
        if len(products) > 1:
            plural_analog_count += 1
            d = evaluator(products)
            if not np.isnan(d):
                product_diversities.append(d)

    product_diversities += [0] * (total - len(product_diversities))

    print(f"Total {plural_analog_count}/{len(product_good_pathways)} query molecules have > 1 generated analogs")
    print(f"Mean product diversity: {np.mean(product_diversities):.3f}")

    # building block diversity
    bb_diversities = []
    for bbs in bb_good_pathways.values():
        bbs = list(bbs)
        if len(bbs) > 1:
            d = evaluator(bbs)
            if not np.isnan(d):
                bb_diversities.append(d)

    bb_diversities += [0] * (total - len(bb_diversities))

    print(f"Mean BB diversity: {np.mean(bb_diversities):.3f}")


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("csv_file")
    parser.add_argument("--total", type=int, default=1000)
    parser.add_argument("--sim_threshold", type=float, default=0.8)
    parser.add_argument("--score_threshold", type=float, default=0.8)

    args = parser.parse_args()

    df = pd.read_csv(args.csv_file)

    # canonicalize
    df["target"] = df["target"].apply(canonicalize)
    df["smiles"] = df["smiles"].apply(canonicalize)

    df = df.dropna(subset=["target", "smiles"])
    df = df.drop_duplicates()

    print(df.loc[df.groupby("target").idxmax()["score"]].select_dtypes(include="number").sum() / args.total)

    evaluator = Evaluator("diversity")

    for thresh in range(50, 100, 5):
        print(f"======== Analogue Threshold {thresh * 0.1} =========")
        calculate_diversity(thresh, evaluator, df, args.total)


if __name__ == "__main__":
    # run: python -m evals.diversity_eval ../expts_temp/enamine_reconstruct/1k_chembl_enamine_reconstruct.csv
    main()
