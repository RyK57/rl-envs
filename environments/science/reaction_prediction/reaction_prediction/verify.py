# /// script
# requires-python = ">=3.10"
# dependencies = ["rdkit==2026.3.6"]
# ///
"""Compare a product SMILES with the recorded product: one JSON line on stdout.

    verify.py GOLD_SMILES ANSWER_SMILES IGNORE_STEREO(0|1)

Atom-map numbers are cleared from both molecules before comparison. An answer that does not parse is
a scoring outcome (`parsed: false`), not an error; the script exits non-zero only on a malformed call
or a gold SMILES RDKit cannot read, so the host can tell an infrastructure failure from a wrong answer.
"""

import json
import sys


def main(argv: list[str]) -> int:
    if len(argv) != 4:
        print("usage: verify.py GOLD_SMILES ANSWER_SMILES IGNORE_STEREO", file=sys.stderr)
        return 2
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import rdFingerprintGenerator, rdMolDescriptors

    RDLogger.DisableLog("rdApp.*")
    gold, answer, ignore_stereo = argv[1], argv[2], argv[3] == "1"

    def read(smiles: str):
        mol = Chem.MolFromSmiles(smiles) if smiles else None
        if mol is not None:
            for atom in mol.GetAtoms():
                atom.SetAtomMapNum(0)
            if ignore_stereo:
                Chem.RemoveStereochemistry(mol)
        return mol

    gold_mol = read(gold)
    if gold_mol is None:
        print(f"gold SMILES does not parse: {gold!r}", file=sys.stderr)
        return 2
    mol = read(answer)
    report = {"parsed": mol is not None, "exact": False, "tanimoto": 0.0, "formula_match": False}
    if mol is not None:
        report["exact"] = Chem.MolToSmiles(mol) == Chem.MolToSmiles(gold_mol)
        generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        fingerprints = generator.GetFingerprint(mol), generator.GetFingerprint(gold_mol)
        report["tanimoto"] = DataStructs.TanimotoSimilarity(*fingerprints)
        report["formula_match"] = rdMolDescriptors.CalcMolFormula(mol) == rdMolDescriptors.CalcMolFormula(gold_mol)
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
