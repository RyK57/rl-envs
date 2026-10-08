# /// script
# requires-python = ">=3.10"
# dependencies = ["rdkit==2026.3.6"]
# ///
"""The computable properties of a structure: one JSON line on stdout.

    props.py SMILES

Exits non-zero only on a malformed call or a SMILES RDKit cannot read, so the host can tell an
infrastructure failure from a wrong answer.
"""

import json
import sys


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: props.py SMILES", file=sys.stderr)
        return 2
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors, rdMolDescriptors

    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(argv[1])
    if mol is None:
        print(f"SMILES does not parse: {argv[1]!r}", file=sys.stderr)
        return 2
    report = {
        "formula": rdMolDescriptors.CalcMolFormula(mol),
        "molecular_weight": round(Descriptors.MolWt(mol), 4),
        "monoisotopic_mass": round(Descriptors.ExactMolWt(mol), 6),
        "heavy_atoms": mol.GetNumHeavyAtoms(),
        "rings": rdMolDescriptors.CalcNumRings(mol),
        "aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(mol),
        "stereocenters": len(Chem.FindMolChiralCenters(mol, includeUnassigned=True, useLegacyImplementation=False)),
    }
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
