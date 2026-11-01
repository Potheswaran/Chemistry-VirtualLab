"""
MoleculeLab — Virtual Chemistry Laboratory
A Streamlit assignment project for exploring chemical structures and molecular properties.
Run locally: streamlit run app.py
"""
from __future__ import annotations

from io import BytesIO
from datetime import datetime
import html
import math

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from rdkit import Chem
from rdkit.Chem import AllChem, Crippen, Descriptors, Draw, Lipinski, rdMolDescriptors
from rdkit.Chem import rdDepictor
import py3Dmol
import streamlit.components.v1 as components

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage

# ───────────────────────── Page setup ─────────────────────────
st.set_page_config(
    page_title="MoleculeLab | Virtual Chemistry",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_NAME = "MoleculeLab"
EXAMPLES = {
    "Ethanol": "CCO",
    "Water": "O",
    "Benzene": "c1ccccc1",
    "Phenol": "Oc1ccccc1",
    "Acetic acid": "CC(=O)O",
    "Aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    "Caffeine": "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
    "Glucose": "C(C1C(C(C(C(O1)O)O)O)O)O",
    "Lactic acid": "CC(O)C(=O)O",
    "Paracetamol": "CC(=O)NC1=CC=C(C=C1)O",
    "Ibuprofen": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
    "Limonene": "CC1=CCC(CC1)C(=C)C",
}

# ───────────────────────── Styling ─────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');
:root { color-scheme: light; }
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #f5f7fb; color: #15233b; }
[data-testid="stHeader"] { background: rgba(245,247,251,.88); }
[data-testid="stSidebar"] { background: #111e32; }
[data-testid="stSidebar"] * { color: #eaf1ff !important; }
[data-testid="stSidebar"] [data-testid="stRadio"] label { padding: 8px 10px; border-radius: 10px; }
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover { background: #20334f; }
.block-container { padding-top: 1.8rem; padding-bottom: 3rem; max-width: 1440px; }
.hero {
    padding: 30px 34px; border-radius: 22px;
    background: linear-gradient(115deg, #10223c 0%, #163d5c 55%, #167d80 100%);
    color: white; margin-bottom: 24px; position: relative; overflow: hidden;
}
.hero:after { content: "⚛"; position: absolute; right: 30px; top: -36px; font-size: 180px; opacity: .08; }
.hero h1 { font-family: 'Space Grotesk', sans-serif; font-size: clamp(2rem, 4vw, 3.2rem); margin: 5px 0 10px; color: white; }
.hero p { color: #d8e7f4; font-size: 1.04rem; max-width: 760px; margin-bottom: 0; }
.eyebrow { text-transform: uppercase; letter-spacing: .18em; font-size: .72rem; font-weight: 700; color: #73e0ce; }
.section-head { font-family: 'Space Grotesk', sans-serif; color: #14243b; font-size: 1.55rem; font-weight: 700; margin: 8px 0 5px; }
.subtle { color: #66758c; font-size: .92rem; }
.metric-card { background: white; border: 1px solid #e4eaf2; border-radius: 16px; padding: 18px 20px; min-height: 112px; box-shadow: 0 5px 20px rgba(24,45,77,.035); }
.metric-label { color: #6b7a90; font-size: .78rem; font-weight: 700; text-transform: uppercase; letter-spacing: .06em; }
.metric-value { font-family: 'Space Grotesk', sans-serif; font-size: 1.8rem; font-weight: 700; color: #152a45; margin-top: 6px; }
.panel { background: white; border: 1px solid #e4eaf2; border-radius: 18px; padding: 22px; box-shadow: 0 5px 20px rgba(24,45,77,.035); }
.pill { display:inline-block; padding: 5px 10px; border-radius: 99px; background:#e5f7f3; color:#087d72; font-size:.76rem; font-weight:700; }
hr { border-color: #e3eaf2; }
.stButton > button[kind="primary"] { background: #0f9387; border-color: #0f9387; }
.stButton > button { border-radius: 10px; font-weight: 600; }
div[data-testid="stDownloadButton"] button { border-radius: 10px; }
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ───────────────────────── Core chemistry helpers ─────────────────────────
def parse_smiles(smiles: str):
    """Return an RDKit molecule or None when the SMILES is invalid."""
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    try:
        return Chem.MolFromSmiles(smiles.strip())
    except (ValueError, TypeError, RuntimeError):
        return None

def molecule_properties(mol) -> dict:
    return {
        "Molecular formula": rdMolDescriptors.CalcMolFormula(mol),
        "Molecular weight (g/mol)": round(Descriptors.MolWt(mol), 3),
        "Exact mass (Da)": round(Descriptors.ExactMolWt(mol), 4),
        "Heavy atoms": int(mol.GetNumHeavyAtoms()),
        "Total atoms (incl. implicit H)": int(mol.GetNumAtoms() + sum(a.GetTotalNumHs() for a in mol.GetAtoms())),
        "Bonds": int(mol.GetNumBonds()),
        "Rotatable bonds": int(Lipinski.NumRotatableBonds(mol)),
        "H-bond donors": int(Lipinski.NumHDonors(mol)),
        "H-bond acceptors": int(Lipinski.NumHAcceptors(mol)),
        "LogP": round(Crippen.MolLogP(mol), 3),
        "Topological polar surface area (Å²)": round(rdMolDescriptors.CalcTPSA(mol), 3),
        "Ring count": int(rdMolDescriptors.CalcNumRings(mol)),
        "Fraction C sp3": round(rdMolDescriptors.CalcFractionCSP3(mol), 3),
    }

def make_2d_png(mol, size=(760, 520)):
    copy = Chem.Mol(mol)
    rdDepictor.Compute2DCoords(copy)
    return Draw.MolToImage(copy, size=size, kekulize=True)

def make_3d_html(mol, style="stick"):
    """Build a self-contained py3Dmol viewer HTML fragment."""
    work = Chem.AddHs(Chem.Mol(mol))
    status = AllChem.EmbedMolecule(work, randomSeed=7)
    if status != 0:
        return None
    try:
        if AllChem.MMFFHasAllMoleculeParams(work):
            AllChem.MMFFOptimizeMolecule(work, maxIters=300)
        elif AllChem.UFFHasAllMoleculeParams(work):
            AllChem.UFFOptimizeMolecule(work, maxIters=300)
    except (ValueError, RuntimeError):
        pass
    block = Chem.MolToMolBlock(work)
    viewer = py3Dmol.view(width=760, height=460)
    viewer.addModel(block, "mol")
    viewer.setStyle({"stick": {}} if style == "stick" else {"sphere": {"scale": 0.28}, "stick": {"radius": 0.12}})
    viewer.setBackgroundColor("#f8fbff")
    viewer.zoomTo()
    return viewer._make_html()

def bond_summary(mol):
    rows = []
    for idx, bond in enumerate(mol.GetBonds(), start=1):
        a1, a2 = bond.GetBeginAtom(), bond.GetEndAtom()
        rows.append({
            "Bond": idx,
            "Atoms": f"{a1.GetSymbol()}{a1.GetIdx()+1} – {a2.GetSymbol()}{a2.GetIdx()+1}",
            "Type": str(bond.GetBondType()).replace("DOUBLE", "Double").replace("TRIPLE", "Triple").replace("SINGLE", "Single").replace("AROMATIC", "Aromatic"),
            "In ring": bool(bond.IsInRing()),
            "Conjugated": bool(bond.GetIsConjugated()),
        })
    return pd.DataFrame(rows)

def create_pdf(molecule_name, smiles, props, observations="Generated with MoleculeLab virtual chemistry laboratory."):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=0.65*inch, leftMargin=0.65*inch, topMargin=0.65*inch, bottomMargin=0.65*inch)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="ReportTitle", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=22, leading=27, textColor=colors.HexColor("#15344d"), alignment=TA_CENTER, spaceAfter=10))
    styles.add(ParagraphStyle(name="SmallBody", parent=styles["BodyText"], fontSize=9, leading=13))
    story = [
        Paragraph("MoleculeLab", styles["ReportTitle"]),
        Paragraph("Virtual Chemistry Laboratory · Molecular Analysis Report", styles["Heading2"]),
        Spacer(1, 8),
        Paragraph(f"<b>Compound:</b> {html.escape(str(molecule_name))}", styles["BodyText"]),
        Paragraph(f"<b>SMILES:</b> {html.escape(str(smiles))}", styles["SmallBody"]),
        Paragraph(f"<b>Generated:</b> {datetime.now().strftime('%d %b %Y, %H:%M')}", styles["SmallBody"]),
        Spacer(1, 14),
        Paragraph("Calculated molecular properties", styles["Heading2"]),
    ]
    table_data = [["Property", "Calculated value"]]
    table_data += [[str(k), str(v)] for k, v in props.items()]
    table = Table(table_data, colWidths=[3.25*inch, 3.1*inch], repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#15344d")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,-1), 9),
        ("GRID", (0,0), (-1,-1), 0.35, colors.HexColor("#d8e1ec")),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f3f7fb")]),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("LEFTPADDING", (0,0), (-1,-1), 8),
        ("RIGHTPADDING", (0,0), (-1,-1), 8),
        ("TOPPADDING", (0,0), (-1,-1), 6),
        ("BOTTOMPADDING", (0,0), (-1,-1), 6),
    ]))
    story.extend([table, Spacer(1, 16), Paragraph("Notes and interpretation", styles["Heading2"]), Paragraph(html.escape(observations).replace("\n", "<br/>"), styles["SmallBody"]), Spacer(1, 18), Paragraph("Educational use only. Results are computational estimates and should be interpreted in context.", styles["Italic"])])
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

def metric(label, value):
    st.markdown(f'<div class="metric-card"><div class="metric-label">{html.escape(str(label))}</div><div class="metric-value">{html.escape(str(value))}</div></div>', unsafe_allow_html=True)

def section(title, subtitle=None):
    st.markdown(f'<div class="section-head">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="subtle">{subtitle}</div>', unsafe_allow_html=True)

def get_active_molecule():
    smiles = st.session_state.get("active_smiles", "CCO")
    return smiles, parse_smiles(smiles)

# ───────────────────────── Sidebar navigation ─────────────────────────
with st.sidebar:
    st.markdown("## 🧪 MoleculeLab")
    st.markdown('<span class="pill">VIRTUAL CHEMISTRY</span>', unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio(
        "LABORATORY MENU",
        ["🏠 Home", "🧬 Molecule Explorer", "📊 Structure Analysis", "⚛️ Bond Lab", "🧠 Learn & Quiz", "📄 Lab Reports", "ℹ️ About"],
        label_visibility="visible",
    )
    st.markdown("---")
    st.caption("An educational cheminformatics project")
    st.caption("Built with Python · Streamlit · RDKit")

# helper needed for download
def _png_bytes(mol):
    buf = BytesIO()
    make_2d_png(mol).save(buf, format="PNG")
    return buf.getvalue()

# ───────────────────────── Home ─────────────────────────
if page == "🏠 Home":
    st.markdown("""
    <div class="hero">
      <div class="eyebrow">DIGITAL SCIENCE · INTERACTIVE LEARNING</div>
      <h1>Explore chemistry<br>at the molecular level.</h1>
      <p>A virtual laboratory to visualize chemical structures, inspect bonds, calculate molecular descriptors, and learn through hands-on exploration.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("### Your laboratory, at a glance")
    c1, c2, c3, c4 = st.columns(4)
    with c1: metric("Example compounds", len(EXAMPLES))
    with c2: metric("Analysis tools", "04")
    with c3: metric("Visualization", "2D + 3D")
    with c4: metric("Report format", "PDF")
    st.write("")
    left, right = st.columns([1.2, 0.8], gap="large")
    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        section("Start a virtual experiment", "Choose a sample compound or enter a SMILES string to begin.")
        choice = st.selectbox("Quick-start compound", list(EXAMPLES.keys()), key="home_example")
        st.caption(f"SMILES: `{EXAMPLES[choice]}`")
        if st.button("Load compound into Molecule Explorer", type="primary", use_container_width=True):
            st.session_state.active_smiles = EXAMPLES[choice]
            st.session_state.active_name = choice
            st.success(f"{choice} loaded. Open Molecule Explorer from the sidebar.")
        st.markdown("</div>", unsafe_allow_html=True)
        st.write("")
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        section("What you can investigate")
        st.markdown("""
        - **Structure:** inspect 2D skeletal formulas and 3D conformations.
        - **Properties:** calculate molecular weight, LogP, TPSA, H-bond counts, and more.
        - **Bonds:** examine bond type, ring membership, and conjugation.
        - **Compare:** explore how molecular descriptors differ between compounds.
        """)
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        section("Suggested first experiment")
        st.markdown("**Compare ethanol and acetic acid**")
        st.write("Both are small organic molecules, but their functional groups and molecular descriptors differ.")
        st.markdown("""
        1. Open **Molecule Explorer** and load Ethanol.
        2. Note the formula, molecular weight, and hydrogen-bond properties.
        3. Open **Structure Analysis** to compare it with Acetic acid.
        4. Export your findings as a laboratory report.
        """)
        st.markdown("</div>", unsafe_allow_html=True)
    st.info("Educational note: molecular descriptors are calculated from the input structure. They do not, by themselves, establish biological activity, toxicity, or experimental outcomes.")

# ───────────────────────── Molecule Explorer ─────────────────────────
elif page == "🧬 Molecule Explorer":
    st.markdown('<div class="hero"><div class="eyebrow">WORKBENCH 01</div><h1>Molecule Explorer</h1><p>Translate a SMILES string into a molecular structure and inspect its computed properties.</p></div>', unsafe_allow_html=True)
    default_name = st.session_state.get("active_name", "Ethanol")
    example_names = list(EXAMPLES.keys())
    selected = st.selectbox("Load a sample compound", ["Custom input"] + example_names, index=(example_names.index(default_name)+1 if default_name in example_names else 1), key="explorer_select")
    if selected != "Custom input":
        st.session_state.active_smiles = EXAMPLES[selected]
        st.session_state.active_name = selected
    with st.form("smiles_form"):
        smiles_input = st.text_input("SMILES notation", value=st.session_state.get("active_smiles", "CCO"), help="SMILES is a text representation of a chemical structure.")
        submitted = st.form_submit_button("Analyze molecule", type="primary")
    if submitted:
        st.session_state.active_smiles = smiles_input.strip()
        st.session_state.active_name = selected if selected != "Custom input" else "Custom molecule"
    smiles, mol = get_active_molecule()
    if mol is None:
        st.error("This SMILES string could not be parsed. Check the notation and try again.")
        st.code("Examples: CCO (ethanol), c1ccccc1 (benzene), CC(=O)O (acetic acid)")
    else:
        props = molecule_properties(mol)
        st.success(f"Structure parsed successfully · {props['Molecular formula']}")
        a, b, c, d = st.columns(4)
        with a: metric("Molecular formula", props["Molecular formula"])
        with b: metric("Molecular weight", f"{props['Molecular weight (g/mol)']} g/mol")
        with c: metric("Heavy atoms", props["Heavy atoms"])
        with d: metric("Rotatable bonds", props["Rotatable bonds"])
        tab2d, tab3d, tabprops = st.tabs(["🧾 2D structure", "🧊 3D model", "📋 Properties"])
        with tab2d:
            left, right = st.columns([1, 1])
            with left:
                st.image(make_2d_png(mol), caption="2D molecular structure", use_container_width=True)
            with right:
                section("Structure summary")
                st.write(f"**Formula:** {props['Molecular formula']}")
                st.write(f"**Atoms:** {props['Total atoms (incl. implicit H)']}")
                st.write(f"**Bonds:** {props['Bonds']}")
                st.write(f"**Rings:** {props['Ring count']}")
                st.write(f"**SMILES:** `{smiles}`")
                st.download_button("Download 2D structure (PNG)", data=make_2d_png(mol).save if False else _png_bytes(mol), file_name="molecule_structure.png", mime="image/png")
        with tab3d:
            view_style = st.radio("Display style", ["stick", "ball-and-stick"], horizontal=True)
            try:
                html_view = make_3d_html(mol, view_style)
                if html_view:
                    components.html(html_view, height=480, scrolling=False)
                    st.caption("3D coordinates are generated computationally and optimized when supported. They are illustrative, not an experimental structure.")
                else:
                    st.warning("A 3D conformer could not be generated for this structure.")
            except Exception as exc:
                st.warning("The 3D viewer could not be prepared for this molecule.")
                st.caption(f"Technical detail: {type(exc).__name__}")
        with tabprops:
            prop_df = pd.DataFrame([{"Property": k, "Value": v} for k, v in props.items()])
            st.dataframe(prop_df, use_container_width=True, hide_index=True)
            st.download_button("Download properties as CSV", prop_df.to_csv(index=False).encode("utf-8"), "molecular_properties.csv", "text/csv")

# ───────────────────────── Structure Analysis ─────────────────────────
elif page == "📊 Structure Analysis":
    st.markdown('<div class="hero"><div class="eyebrow">WORKBENCH 02</div><h1>Structure–Property Analysis</h1><p>Compare molecular descriptors across compounds and explore structure–property patterns.</p></div>', unsafe_allow_html=True)
    selected_names = st.multiselect("Choose compounds to compare", list(EXAMPLES.keys()), default=["Ethanol", "Acetic acid", "Benzene"])
    if selected_names:
        rows = []
        for name in selected_names:
            mol = parse_smiles(EXAMPLES[name])
            if mol:
                rows.append({"Compound": name, "SMILES": EXAMPLES[name], **molecule_properties(mol)})
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)
        numeric_cols = [c for c in df.select_dtypes(include="number").columns if c not in ("Total atoms (incl. implicit H)",)]
        if numeric_cols:
            chosen_prop = st.selectbox("Descriptor to visualize", numeric_cols, index=numeric_cols.index("Molecular weight (g/mol)") if "Molecular weight (g/mol)" in numeric_cols else 0)
            fig, ax = plt.subplots(figsize=(9, 4.3))
            ax.bar(df["Compound"], df[chosen_prop])
            ax.set_ylabel(chosen_prop)
            ax.set_title(f"{chosen_prop} by compound")
            ax.tick_params(axis="x", rotation=25)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download comparison CSV", csv, "structure_property_comparison.csv", "text/csv")
        st.caption("Descriptors are computed from SMILES using RDKit. Correlation between descriptors does not prove causation.")
    else:
        st.info("Select at least one compound to display a comparison.")

# ───────────────────────── Bond Lab ─────────────────────────
elif page == "⚛️ Bond Lab":
    st.markdown('<div class="hero"><div class="eyebrow">WORKBENCH 03</div><h1>Bond Lab</h1><p>Inspect the bond inventory of a molecule, including bond order, ring membership, and conjugation flags.</p></div>', unsafe_allow_html=True)
    smiles, mol = get_active_molecule()
    with st.expander("Choose a molecule for bond analysis", expanded=True):
        sample = st.selectbox("Sample compound", list(EXAMPLES.keys()), index=0, key="bond_example")
        if st.button("Use selected compound"):
            st.session_state.active_smiles = EXAMPLES[sample]
            st.session_state.active_name = sample
            st.rerun()
        bond_smiles = st.text_input("Or enter SMILES", value=smiles, key="bond_smiles")
    bond_mol = parse_smiles(bond_smiles)
    if bond_mol is None:
        st.error("Invalid SMILES. Please check the structure.")
    else:
        bonds = bond_summary(bond_mol)
        col1, col2, col3 = st.columns(3)
        with col1: metric("Total bonds", len(bonds))
        with col2: metric("Ring bonds", int(bonds["In ring"].sum()) if not bonds.empty else 0)
        with col3: metric("Conjugated bonds", int(bonds["Conjugated"].sum()) if not bonds.empty else 0)
        if bonds.empty:
            st.info("This molecule has no bonds (for example, a single atom).")
        else:
            st.dataframe(bonds, use_container_width=True, hide_index=True)
            st.download_button("Download bond inventory", bonds.to_csv(index=False).encode("utf-8"), "bond_inventory.csv", "text/csv")
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        section("How to read this table")
        st.markdown("""
        - **Single, double, triple:** bond order assigned in the molecular graph.
        - **Aromatic:** bond is part of an aromatic system as represented by RDKit.
        - **In ring:** the bond belongs to at least one ring in the molecular graph.
        - **Conjugated:** the bond is flagged as part of a conjugated system.

        This table describes the encoded structure. It is not a quantum-chemical calculation of bond energies or measured bond lengths.
        """)
        st.markdown("</div>", unsafe_allow_html=True)

# ───────────────────────── Learn & Quiz ─────────────────────────
elif page == "🧠 Learn & Quiz":
    st.markdown('<div class="hero"><div class="eyebrow">LEARNING STUDIO</div><h1>Learn & Quiz</h1><p>Review cheminformatics basics, then test your understanding with a short quiz.</p></div>', unsafe_allow_html=True)
    tab_learn, tab_quiz = st.tabs(["📚 Quick guide", "🎯 Knowledge check"])
    with tab_learn:
        concepts = [
            ("SMILES", "A line notation that encodes atoms, bonds, branches, and ring closures in a molecule."),
            ("Molecular formula", "A count of the element types present in the molecular graph, including implicit hydrogens as calculated by RDKit."),
            ("Molecular weight", "The sum of atomic weights represented by a molecule; the reported value depends on the atomic-weight convention."),
            ("LogP", "A computed estimate of partitioning between octanol and water, often used as a descriptor of lipophilicity."),
            ("TPSA", "Topological polar surface area, a 2D descriptor related to polar atoms and their bonding environments."),
            ("Hydrogen-bond donors / acceptors", "Rule-based counts of groups that can donate or accept hydrogen bonds under the descriptor model."),
            ("Conformation", "A 3D arrangement of atoms accessible through rotations and geometry changes without changing connectivity."),
        ]
        for title, description in concepts:
            with st.expander(title):
                st.write(description)
    with tab_quiz:
        questions = [
            {"q": "Which notation is used in this app to enter molecular structures?", "options": ["SMILES", "HTML", "CSV only", "JPEG"], "answer": "SMILES", "why": "SMILES encodes molecular connectivity in text form."},
            {"q": "Which tool in this project calculates cheminformatics descriptors?", "options": ["RDKit", "PowerPoint", "Flask only", "CSS"], "answer": "RDKit", "why": "RDKit provides molecular parsing, descriptors, and structure drawing."},
            {"q": "What does a 3D conformer generated by this app represent?", "options": ["An illustrative computed geometry", "A guaranteed crystal structure", "A lab measurement", "A reaction yield"], "answer": "An illustrative computed geometry", "why": "The app embeds and optionally optimizes a computational conformer; it is not a measured structure."},
            {"q": "What does LogP generally describe?", "options": ["Estimated lipophilicity", "Number of rings only", "Bond length", "Boiling point measurement"], "answer": "Estimated lipophilicity", "why": "LogP estimates partitioning between octanol and water."},
            {"q": "Does a higher descriptor value alone prove a compound is more effective as a medicine?", "options": ["No", "Yes, always", "Only for aromatic compounds", "Only if the molecule is large"], "answer": "No", "why": "Descriptors are useful features, but they do not establish clinical efficacy or safety."},
        ]
        if "quiz_answers" not in st.session_state:
            st.session_state.quiz_answers = {}
        with st.form("quiz_form"):
            for i, q in enumerate(questions):
                st.markdown(f"**{i+1}. {q['q']}**")
                st.radio("Select one answer", q["options"], key=f"quiz_{i}", label_visibility="collapsed")
            quiz_submit = st.form_submit_button("Check my answers", type="primary")
        if quiz_submit:
            score = 0
            for i, q in enumerate(questions):
                selected_answer = st.session_state.get(f"quiz_{i}")
                if selected_answer == q["answer"]:
                    score += 1
            st.session_state.quiz_score = score
            st.session_state.quiz_checked = True
        if st.session_state.get("quiz_checked"):
            score = st.session_state.quiz_score
            st.metric("Your score", f"{score} / {len(questions)}", f"{round(score/len(questions)*100)}%")
            for i, q in enumerate(questions):
                correct = st.session_state.get(f"quiz_{i}") == q["answer"]
                st.markdown(f"{'✅' if correct else '❌'} **Q{i+1}:** {q['why']}")
            if st.button("Try again"):
                st.session_state.quiz_checked = False
                st.rerun()

# ───────────────────────── Reports ─────────────────────────
elif page == "📄 Lab Reports":
    st.markdown('<div class="hero"><div class="eyebrow">DOCUMENTATION</div><h1>Laboratory Reports</h1><p>Generate a downloadable PDF with the molecule identity and calculated descriptor table.</p></div>', unsafe_allow_html=True)
    smiles, mol = get_active_molecule()
    if mol is None:
        st.warning("Load a valid molecule in Molecule Explorer before generating a report.")
    else:
        props = molecule_properties(mol)
        col1, col2 = st.columns([1, 1])
        with col1:
            name = st.text_input("Compound name", value=st.session_state.get("active_name", "Molecule"))
            st.text_input("SMILES", value=smiles, disabled=True)
        with col2:
            notes = st.text_area("Observation / interpretation", value="Calculated descriptors summarize selected structural features. These values are computational estimates, not experimental measurements.", height=135)
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        section("Report preview")
        st.write(f"**Compound:** {name}")
        st.write(f"**Formula:** {props['Molecular formula']}")
        st.write(f"**Molecular weight:** {props['Molecular weight (g/mol)']} g/mol")
        st.write(f"**Descriptor count:** {len(props)}")
        st.markdown("</div>", unsafe_allow_html=True)
        pdf = create_pdf(name, smiles, props, notes)
        st.download_button("⬇️ Download PDF laboratory report", pdf, file_name=f"{re.sub(r'[^A-Za-z0-9_-]+', '_', name).strip('_') or 'molecule'}_lab_report.pdf", mime="application/pdf", type="primary")

# ───────────────────────── About ─────────────────────────
elif page == "ℹ️ About":
    st.markdown('<div class="hero"><div class="eyebrow">PROJECT INFORMATION</div><h1>About MoleculeLab</h1><p>A student-friendly cheminformatics virtual laboratory designed to make molecular structure and descriptor analysis more accessible.</p></div>', unsafe_allow_html=True)
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    section("Project objectives")
    st.markdown("""
    MoleculeLab demonstrates how computational chemistry tools can support introductory molecular exploration. It accepts SMILES notation, parses chemical structures, visualizes molecules, computes common molecular descriptors, inventories bonds, compares compounds, and generates a PDF report.

    **Technology stack**
    - **Streamlit:** interactive web interface and navigation.
    - **RDKit:** molecular parsing, 2D drawing, conformer generation, and descriptor calculation.
    - **py3Dmol:** browser-based 3D molecule viewer.
    - **Pandas / Matplotlib:** data tables and comparison charts.
    - **ReportLab:** downloadable PDF report generation.
    """)
    st.markdown("</div>", unsafe_allow_html=True)
    st.warning("Scope and limitations: this is an educational application. It does not replace laboratory experiments, professional computational chemistry, toxicology review, or clinical evaluation. It does not currently include chemical reaction simulation or a persistent user database.")
    st.caption("MoleculeLab · Virtual Chemistry Laboratory · Built for educational demonstration")
