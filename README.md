# MoleculeLab — Virtual Chemistry Laboratory

MoleculeLab is a Streamlit-based virtual chemistry laboratory for exploring molecular structures and basic cheminformatics descriptors. It is designed as an educational project and can be deployed on Streamlit Community Cloud from a GitHub repository.

## Features

- **Redesigned Home dashboard** with a guided workflow, feature tiles, descriptor highlights, and quick-start compounds.
- **Molecule Explorer** to enter SMILES, view a 2D structure, inspect a 3D conformer, calculate molecular descriptors, and read a built-in descriptor interpretation guide.
- **Structure–Property Analysis** to compare compounds, visualize a selected descriptor, and learn how to interpret comparisons.
- **Bond Lab** to inspect bond order, ring membership, and conjugation flags.
- **Learn & Quiz** with a short concept guide and interactive knowledge check.
- **Project overview** explaining the input → parsing → calculation → presentation workflow and the role of each technology.
- **Lab Reports** to export calculated molecular properties to a PDF.
- **CSV downloads** for property tables, bond inventories, and comparisons.

## Technology stack

Python · Streamlit · RDKit · py3Dmol · Pandas · Matplotlib · ReportLab

## Run locally

Use Python 3.12 in a virtual environment for the tested deployment configuration.

```bash
python -m venv .venv
```

Activate the environment:

**Windows PowerShell**
```powershell
.venv\Scripts\Activate.ps1
```

**macOS / Linux**
```bash
source .venv/bin/activate
```

Install dependencies and start the app:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

The app will open in your browser at the local address printed by Streamlit.

## Deploy to GitHub + Streamlit Community Cloud

1. Create a new repository on GitHub, for example `moleculelab-virtual-chemistry`.
2. Upload `app.py`, `requirements.txt`, `README.md`, and `.gitignore` to the repository root.
3. Open Streamlit Community Cloud and sign in with GitHub.
4. Choose **Create app**, select the repository and branch, and set the main file path to `app.py`.
5. Deploy. Streamlit will install the packages listed in `requirements.txt`.

Keep the repository public if your assignment requires a public source link. Never commit passwords, API keys, or other secrets.

## Suggested repository structure

```text
moleculelab-virtual-chemistry/
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Using SMILES

SMILES is a text notation for molecular connectivity. A few examples included in the app:

| Compound | SMILES |
|---|---|
| Ethanol | `CCO` |
| Water | `O` |
| Benzene | `c1ccccc1` |
| Phenol | `Oc1ccccc1` |
| Acetic acid | `CC(=O)O` |
| Aspirin | `CC(=O)Oc1ccccc1C(=O)O` |

## Scientific limitations

- Descriptors depend on the encoded molecular structure and RDKit's descriptor definitions.
- The 3D viewer displays a generated computational conformer, not a verified experimental crystal structure.
- Descriptor values alone do not establish toxicity, efficacy, bioavailability, or clinical safety.
- The project is educational software, not a substitute for professional chemistry tools or laboratory safety procedures.

## Assignment presentation notes

You can describe this project as a browser-based virtual chemistry laboratory that combines molecular visualization, descriptor calculation, compound comparison, bond inspection, knowledge assessment, and PDF reporting in one interface.

## Custom SMILES and 3D analysis

Open **Molecule Explorer** and choose **Custom input** (or simply replace the SMILES field). Enter a molecule name if desired, paste a SMILES string, and select **Analyze my SMILES**. The app validates the notation with RDKit and provides:

- A 2D molecular structure drawing.
- An interactive 3D conformer with stick, ball-and-stick, and space-filling display styles.
- Optional labels for non-hydrogen atoms.
- A downloadable 3D SDF structure.
- Calculated molecular formula, molecular weight, exact mass, atom and bond counts, LogP, TPSA, hydrogen-bond donor/acceptor counts, ring count, and related descriptors.
- CSV export of the descriptor table.

In **Structure Analysis**, add custom compounds in the format `Name | SMILES`, one per line, to compare their calculated descriptors with sample molecules. Invalid SMILES lines are flagged and excluded from the comparison.

### Example SMILES to try

| Compound | SMILES |
|---|---|
| Ethanol | `CCO` |
| Benzene | `c1ccccc1` |
| Aspirin | `CC(=O)Oc1ccccc1C(=O)O` |
| Caffeine | `Cn1c(=O)c2c(ncn2C)n(C)c1=O` |
| Glucose | `C(C1C(C(C(C(O1)O)O)O)O)O` |

**Scientific note:** the 3D model is a generated and, when supported, force-field-optimized conformer. It is not an experimentally measured crystal structure. Descriptor outputs are computational estimates and do not establish biological activity, safety, or clinical effectiveness.
