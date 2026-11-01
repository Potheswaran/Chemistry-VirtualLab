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
import re

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
    initial_sidebar_state="collapsed",
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
:root { color-scheme: dark; --black: #000000; --panel: #090909; --yellow: #ffd600; --yellow-soft: #fff176; }
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] { background: var(--black) !important; color: var(--yellow) !important; }
[data-testid="stHeader"] { background: rgba(0,0,0,.96) !important; }
[data-testid="stSidebar"] { display: none; }
.block-container { padding-top: 1.8rem; padding-bottom: 3rem; max-width: 1440px; }
.stApp p, .stApp li, .stApp label, .stApp span, .stApp div, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5 { color: var(--yellow); }
.top-brand { font-family: 'Space Grotesk', sans-serif; font-size: 1.35rem; font-weight: 700; color: var(--yellow) !important; padding: .42rem 0; white-space: nowrap; }
.top-brand span { color: var(--yellow-soft) !important; }
.st-key-top_nav_bar { position: sticky; top: 4.2rem; z-index: 99999; background: #000 !important; padding: .65rem .5rem .8rem; border-bottom: 2px solid var(--yellow); margin-bottom: 1.4rem; box-shadow: 0 8px 14px rgba(0,0,0,.9); }
.st-key-top_nav_bar [data-testid="stRadio"] > div { gap: .3rem; flex-wrap: wrap; }
.st-key-top_nav_bar [data-testid="stRadio"] label { border: 1px solid var(--yellow) !important; background: #080808 !important; color: var(--yellow) !important; padding: .42rem .72rem; border-radius: 999px; margin: 0; font-size: .84rem; opacity: 1 !important; }\n.st-key-top_nav_bar [data-testid="stRadio"] label p, .st-key-top_nav_bar [data-testid="stRadio"] label span, .st-key-top_nav_bar [data-testid="stRadio"] label div { color: var(--yellow) !important; opacity: 1 !important; }
.st-key-top_nav_bar [data-testid="stRadio"] label:has(input:checked) { background: var(--yellow); border-color: var(--yellow); color: #000 !important; }
.st-key-top_nav_bar [data-testid="stRadio"] label:has(input:checked) p, .st-key-top_nav_bar [data-testid="stRadio"] label:has(input:checked) span, .st-key-top_nav_bar [data-testid="stRadio"] label:has(input:checked) div { color: #000 !important; opacity: 1 !important; }
.st-key-top_nav_bar [data-testid="stRadio"] label p { margin: 0; }
.st-key-top_navigation_radio { position: static; }
.hero { padding: 30px 34px; border-radius: 18px; background: #050505; border: 1px solid var(--yellow); color: var(--yellow); margin-bottom: 24px; position: relative; overflow: hidden; }
.hero:after { content: "⚛"; position: absolute; right: 30px; top: -36px; font-size: 180px; color: var(--yellow); opacity: .09; }
.hero h1 { font-family: 'Space Grotesk', sans-serif; font-size: clamp(2rem, 4vw, 3.2rem); margin: 5px 0 10px; color: var(--yellow) !important; }
.hero p { color: var(--yellow-soft) !important; font-size: 1.04rem; max-width: 760px; margin-bottom: 0; }
.eyebrow { text-transform: uppercase; letter-spacing: .18em; font-size: .72rem; font-weight: 700; color: var(--yellow) !important; }
.section-head { font-family: 'Space Grotesk', sans-serif; color: var(--yellow) !important; font-size: 1.55rem; font-weight: 700; margin: 8px 0 5px; }
.subtle, [data-testid="stCaptionContainer"] { color: var(--yellow-soft) !important; font-size: .92rem; }
.metric-card { background: #080808; border: 1px solid var(--yellow); border-radius: 12px; padding: 18px 20px; min-height: 112px; box-shadow: none; }
.metric-label { color: var(--yellow-soft) !important; font-size: .78rem; font-weight: 700; text-transform: uppercase; letter-spacing: .06em; }
.metric-value { font-family: 'Space Grotesk', sans-serif; font-size: 1.8rem; font-weight: 700; color: var(--yellow) !important; margin-top: 6px; }
.panel, [data-testid="stExpander"], [data-testid="stVerticalBlockBorderWrapper"] { background: #050505; border: 1px solid #6b5b00; border-radius: 12px; box-shadow: none; }
.pill { display:inline-block; padding: 5px 10px; border-radius: 99px; background: var(--yellow); color:#000 !important; font-size:.76rem; font-weight:700; }
hr { border-color: var(--yellow) !important; }
.stButton > button, div[data-testid="stDownloadButton"] button, [data-testid="stFormSubmitButton"] button { border: 1px solid var(--yellow) !important; background: #080808 !important; color: var(--yellow) !important; border-radius: 9px; font-weight: 700; }
.stButton > button[kind="primary"], [data-testid="stFormSubmitButton"] button[kind="primary"] { background: var(--yellow) !important; border-color: var(--yellow) !important; color: #000 !important; }
.stButton > button:hover, div[data-testid="stDownloadButton"] button:hover { background: var(--yellow) !important; color: #000 !important; }
input, textarea, [data-baseweb="input"], [data-baseweb="select"] > div, [data-baseweb="textarea"] { background: #080808 !important; color: var(--yellow) !important; border-color: var(--yellow) !important; }
[data-baseweb="select"] *, [role="listbox"], [role="option"] { background-color: #080808 !important; color: var(--yellow) !important; }
[data-testid="stDataFrame"], [data-testid="stTable"] { border: 1px solid var(--yellow); }
[data-testid="stTabs"] button { color: var(--yellow) !important; }
[data-testid="stTabs"] button[aria-selected="true"] { border-bottom-color: var(--yellow) !important; }
[data-testid="stAlert"] { background: #090909 !important; border: 1px solid var(--yellow) !important; color: var(--yellow) !important; }
code, pre { background: #111 !important; color: var(--yellow) !important; }
/* Strong contrast overrides for navigation and all interactive controls */
/* Primary yellow buttons must always use black text, including nested Streamlit spans and paragraphs. */
.stApp button[kind="primary"],
.stApp [data-testid="stFormSubmitButton"] button,
.stApp [data-testid="stDownloadButton"] button[kind="primary"] {
    background: #ffd600 !important;
    border-color: #ffd600 !important;
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
}
.stApp button[kind="primary"] *,
.stApp [data-testid="stFormSubmitButton"] button *,
.stApp [data-testid="stDownloadButton"] button[kind="primary"] * {
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
    opacity: 1 !important;
}
.stApp button[kind="primary"]:hover,
.stApp [data-testid="stFormSubmitButton"] button:hover,
.stApp [data-testid="stDownloadButton"] button[kind="primary"]:hover {
    background: #ffe766 !important;
    color: #000000 !important;
}
.stApp button[kind="primary"]:hover *,
.stApp [data-testid="stFormSubmitButton"] button:hover * {
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
}
.st-key-top_nav_bar [data-testid="stRadio"] label:not(:has(input:checked)) * { color: #ffd600 !important; }
.st-key-top_nav_bar [data-testid="stRadio"] label:has(input:checked) { background: #ffd600 !important; }
.st-key-top_nav_bar [data-testid="stRadio"] label:has(input:checked) * { color: #000000 !important; }
[data-baseweb="select"] > div, [data-baseweb="input"] > div, [data-baseweb="textarea"] { background-color: #080808 !important; }
[data-baseweb="select"] input, [data-baseweb="input"] input, textarea { color: #ffd600 !important; -webkit-text-fill-color: #ffd600 !important; }
[data-testid="stSelectbox"] svg { fill: #ffd600 !important; }
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
    viewer.setBackgroundColor("#000000")
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
        Paragraph(f"<b>Generated:</b> {datetime.now().strftime('%d %b %Y')}", styles["SmallBody"]),
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

def create_batch_pdf(student_name, registration_number, compounds, observations):
    """Create one PDF containing student details and property tables for many compounds."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=0.6*inch, leftMargin=0.6*inch, topMargin=0.55*inch, bottomMargin=0.55*inch)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="BatchTitle", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=21, leading=25, textColor=colors.HexColor("#b38b00"), alignment=TA_CENTER, spaceAfter=10))
    styles.add(ParagraphStyle(name="BatchSmall", parent=styles["BodyText"], fontSize=8.5, leading=11))
    story = [Paragraph("MoleculeLab", styles["BatchTitle"]), Paragraph("Virtual Chemistry Laboratory · Multi-Compound Analysis Report", styles["Heading2"]), Spacer(1, 10),
             Paragraph(f"<b>Student name:</b> {html.escape(student_name)}", styles["BodyText"]),
             Paragraph(f"<b>Registration number:</b> {html.escape(registration_number)}", styles["BodyText"]),
             Paragraph(f"<b>Generated:</b> {datetime.now().strftime('%d %b %Y')}", styles["BatchSmall"]), Spacer(1, 12),
             Paragraph(f"<b>Number of compounds:</b> {len(compounds)}", styles["BodyText"]), Spacer(1, 10)]
    for idx, item in enumerate(compounds, 1):
        if idx > 1:
            story.append(Spacer(1, 12))
        story.extend([Paragraph(f"{idx}. {html.escape(item['name'])}", styles["Heading2"]),
                      Paragraph(f"<b>SMILES:</b> {html.escape(item['smiles'])}", styles["BatchSmall"]),
                      Paragraph(f"<b>Molecular formula:</b> {html.escape(str(item['props'].get('Molecular formula', 'N/A')))} &nbsp;&nbsp; <b>Molecular weight:</b> {html.escape(str(item['props'].get('Molecular weight (g/mol)', 'N/A')))} g/mol", styles["BatchSmall"]), Spacer(1, 5)])
        table_data = [["Property", "Calculated value"]] + [[str(k), str(v)] for k, v in item['props'].items()]
        table = Table(table_data, colWidths=[3.15*inch, 3.0*inch], repeatRows=1)
        table.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#ffd600")),("TEXTCOLOR",(0,0),(-1,0),colors.black),("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("FONTSIZE",(0,0),(-1,-1),8),("GRID",(0,0),(-1,-1),0.35,colors.HexColor("#b38b00")),("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#fff9d6")]),("VALIGN",(0,0),(-1,-1),"TOP"),("LEFTPADDING",(0,0),(-1,-1),6),("RIGHTPADDING",(0,0),(-1,-1),6),("TOPPADDING",(0,0),(-1,-1),4),("BOTTOMPADDING",(0,0),(-1,-1),4)]))
        story.extend([table, Spacer(1, 6)])
    story.extend([Paragraph("Observations / interpretation", styles["Heading2"]), Paragraph(html.escape(observations).replace("\n", "<br/>"), styles["BatchSmall"]), Spacer(1, 10), Paragraph("Educational use only. Results are computational estimates and should not be treated as experimental measurements.", styles["Italic"])])
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

# ───────────────────────── Top navigation ─────────────────────────
NAV_ITEMS = {
    "Home": "🏠 Home",
    "Explore": "🧬 Molecule Explorer",
    "Analysis": "📊 Structure Analysis",
    "Bond Lab": "⚛️ Bond Lab",
    "Learn & Quiz": "🧠 Learn & Quiz",
    "Reports": "📄 Lab Reports",
    "About": "ℹ️ About",
}
with st.container(key="top_nav_bar"):
    brand_col, nav_col = st.columns([1.05, 5.95], vertical_alignment="center")
    with brand_col:
        st.markdown('<div class="top-brand">🧪 Molecule<span>Lab</span></div>', unsafe_allow_html=True)
    with nav_col:
        selected_nav = st.radio(
            "Main navigation",
            list(NAV_ITEMS.keys()),
            horizontal=True,
            label_visibility="collapsed",
            key="top_navigation_radio",
        )
page = NAV_ITEMS[selected_nav]
st.markdown("<div style='height: .35rem'></div>", unsafe_allow_html=True)

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
    st.markdown("### Start a virtual experiment")
    st.markdown("Choose a sample compound to load into the explorer, or navigate directly to a tool using the top menu.")
    quick_col, action_col = st.columns([2, 1], gap="medium", vertical_alignment="bottom")
    with quick_col:
        choice = st.selectbox("Sample compound", list(EXAMPLES.keys()), key="home_example")
        st.caption(f"SMILES: `{EXAMPLES[choice]}`")
    with action_col:
        if st.button("Load compound", type="primary", use_container_width=True):
            st.session_state.active_smiles = EXAMPLES[choice]
            st.session_state.active_name = choice
            st.success(f"{choice} loaded. Open **Explore** from the top navigation.")
    st.markdown("### Explore the laboratory")
    st.markdown("""
    - **Molecule Explorer** — view 2D structures, 3D conformations, and molecular descriptors.
    - **Structure Analysis** — compare compounds and visualize property differences.
    - **Bond Lab** — inspect bond order, ring membership, and conjugation.
    - **Learn & Quiz** — review cheminformatics concepts and test your knowledge.
    - **Lab Reports** — export molecular analysis as a PDF.
    """)
    st.caption("Educational use only: calculated descriptors are not experimental measurements and do not establish biological activity, toxicity, or clinical outcomes.")

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
            fig, ax = plt.subplots(figsize=(9, 4.3), facecolor="#000000")
            ax.set_facecolor("#000000")
            ax.bar(df["Compound"], df[chosen_prop], color="#ffd600", edgecolor="#fff176")
            ax.set_ylabel(chosen_prop, color="#ffd600")
            ax.set_title(f"{chosen_prop} by compound", color="#ffd600")
            ax.tick_params(axis="x", rotation=25, colors="#ffd600")
            ax.tick_params(axis="y", colors="#ffd600")
            for spine in ax.spines.values():
                spine.set_color("#ffd600")
            ax.grid(axis="y", color="#665500", alpha=0.5)
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
    st.markdown('<div class="hero"><div class="eyebrow">DOCUMENTATION</div><h1>Laboratory Reports</h1><p>Enter student details and add as many compounds as needed to create one downloadable PDF report.</p></div>', unsafe_allow_html=True)
    student_col, reg_col = st.columns(2)
    with student_col:
        student_name = st.text_input("Student full name", placeholder="Enter your name", key="report_student_name")
    with reg_col:
        registration_number = st.text_input("Registration number", placeholder="Enter your registration number", key="report_registration_number")
    st.markdown("### Compound list")
    st.caption("Enter one compound per line in the format: Compound name | SMILES. Add as many lines as you need (up to 50 compounds per report).")
    default_compounds = "Ethanol | CCO\nBenzene | c1ccccc1\nAspirin | CC(=O)Oc1ccccc1C(=O)O"
    compound_text = st.text_area("Compounds (name | SMILES)", value=st.session_state.get("report_compound_text", default_compounds), height=190, key="report_compound_text", help="Example: Ethanol | CCO")
    notes = st.text_area("Observation / interpretation", value="Calculated descriptors summarize selected structural features. These values are computational estimates, not experimental measurements.", height=110, key="report_observations")
    if st.button("🔬 Validate compounds", type="primary"):
        st.session_state["report_validate_requested"] = True
    compounds = []
    errors = []
    lines = [line.strip() for line in compound_text.splitlines() if line.strip()]
    if len(lines) > 50:
        errors.append("Please limit each report to 50 compounds or fewer.")
    else:
        for line_no, line in enumerate(lines, 1):
            if "|" not in line:
                errors.append(f"Line {line_no}: use the format Compound name | SMILES.")
                continue
            compound_name, compound_smiles = [part.strip() for part in line.split("|", 1)]
            mol = parse_smiles(compound_smiles)
            if not compound_name or not compound_smiles:
                errors.append(f"Line {line_no}: both compound name and SMILES are required.")
            elif mol is None:
                errors.append(f"Line {line_no} ({compound_name}): invalid SMILES string.")
            else:
                compounds.append({"name": compound_name, "smiles": compound_smiles, "props": molecule_properties(mol)})
    if st.session_state.get("report_validate_requested"):
        if errors:
            for err in errors:
                st.error(err)
        else:
            st.success(f"Validated {len(compounds)} compound(s) successfully.")
    if compounds and not errors:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        section("Report preview")
        st.write(f"**Student:** {student_name or 'Not entered'}  ·  **Registration number:** {registration_number or 'Not entered'}")
        st.write(f"**Compounds included:** {len(compounds)}")
        st.dataframe(pd.DataFrame([{"Compound": c["name"], "SMILES": c["smiles"], "Formula": c["props"].get("Molecular formula", "N/A"), "Molecular weight (g/mol)": c["props"].get("Molecular weight (g/mol)", "N/A")} for c in compounds]), use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)
        if st.button("📄 Generate student laboratory report", type="primary"):
            if not student_name.strip() or not registration_number.strip():
                st.error("Please enter both your full name and registration number before generating the report.")
            else:
                pdf = create_batch_pdf(student_name.strip(), registration_number.strip(), compounds, notes)
                safe_student = re.sub(r"[^A-Za-z0-9_-]+", "_", student_name).strip("_") or "student"
                st.download_button("⬇️ Download PDF laboratory report", pdf, file_name=f"{safe_student}_chemistry_lab_report.pdf", mime="application/pdf", type="primary")

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
