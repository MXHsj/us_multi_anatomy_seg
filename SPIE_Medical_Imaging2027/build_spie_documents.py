from __future__ import annotations

import re
from pathlib import Path

from docx import Document
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "SPIE_MedicalImaging2026"
FIG = ROOT / "ASMUS2026_Evaluating_Ultrasound_Segmentation_Models_MONAI_US_WG" / "figures"

TITLE = (
    "When Do Promptable Foundation Models Simplify Ultrasound Segmentation? "
    "A Multi-Anatomy Benchmark and Prompt-Robustness Study"
)

AUTHORS = (
    "Xihan Ma; Alvin Kimbowa; Gautham Krishnan P; Luohong Wu; Songlin Hou; "
    "Tejas Haritsa V K; Gabriella d'Albenzio; Chunna Yang; Qi Wang; Zhe Min; "
    "Matthias Seibold; Philipp Fürnstahl; Haichong K. Zhang; Tina Kapur; "
    "Stephen R. Aylward; Ilker Hacihaliloglu"
)

TECHNICAL_ABSTRACT = (
    "Promptable foundation models could reduce task-specific development for ultrasound segmentation, "
    "but their reliability depends on the information supplied by the prompt and on target morphology. "
    "We performed a model-centered technology assessment using 13 public two-dimensional B-mode ultrasound "
    "datasets comprising 120,313 images, 118,852 positive masks, and 11 anatomical categories. Five released "
    "models (MedSAM, SAMUS, UltraSAM, MedSAM3, and Medical SAM3) were evaluated without additional tuning using "
    "box, single-point, label-derived text, generic text, and text-plus-box prompts in a common registry-decoder "
    "pipeline. To avoid conflating model capability with prompt information, results were organized within prompt "
    "families rather than as one cross-modal leaderboard. In the seven-dataset cross-model reporting subset, the "
    "macro-average Dice scores were 72.73% and 85.31% for box-prompted MedSAM and UltraSAM; 40.53% and 71.21% for "
    "point-prompted SAMUS and UltraSAM; and 39.07% and 26.14% for label-prompted MedSAM3 and Medical SAM3, "
    "respectively. Within MedSAM3, adding an oracle box to the label prompt increased macro-average Dice from "
    "39.07% to 61.06%. Compact, well-circumscribed targets were generally easier than thin, elongated, or "
    "weak-boundary structures. In a descriptive UltraSAM analysis across all 13 datasets, translating a box by "
    "10% of its dimensions reduced Dice by about five percentage points on average, more than comparable scaling; "
    "text specificity mattered most when competing structures were present. These point estimates characterize "
    "released-checkpoint behavior, not statistical superiority or strict unseen-data generalization, because prompt "
    "interfaces differ and some training corpora overlap at the dataset level. Prompt information, target geometry, "
    "and provenance should therefore be treated as first-order variables when assessing ultrasound segmentation "
    "technology for downstream workflow use."
)

PROGRAM_SUMMARY = (
    "We assess five promptable segmentation foundation models across 13 public B-mode ultrasound datasets and "
    "11 anatomical categories. Prompt-family comparisons show that performance depends strongly on the supplied "
    "localization information, target geometry, and wording specificity. Compact targets are segmented more "
    "reliably than thin or weak-boundary structures, and small box translations can cause appreciable degradation. "
    "The study provides a reproducible, multi-anatomy framework for technology assessment while explicitly "
    "separating prompt effects from model effects and documenting dataset-provenance and statistical limitations."
)

KEYWORDS = (
    "medical ultrasound; image segmentation; foundation models; technology assessment; "
    "prompt robustness; image-based ground truth; failure analysis; artificial intelligence"
)


def count_words(text: str) -> int:
    return len(re.findall(r"\b[\w%+-]+(?:['’-][\w%+-]+)*\b", text))


def set_cell_margins(cell, top=80, start=110, bottom=80, end=110):
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for edge, value in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = tc_mar.find(qn(f"w:{edge}"))
        if node is None:
            node = OxmlElement(f"w:{edge}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def set_cell_width(cell, width_dxa: int):
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_w = tc_pr.first_child_found_in("w:tcW")
    if tc_w is None:
        tc_w = OxmlElement("w:tcW")
        tc_pr.append(tc_w)
    tc_w.set(qn("w:w"), str(width_dxa))
    tc_w.set(qn("w:type"), "dxa")


def set_table_geometry(table, widths_dxa: list[int], indent_dxa: int = 110):
    table.autofit = False
    tbl_pr = table._tbl.tblPr
    tbl_w = tbl_pr.first_child_found_in("w:tblW")
    if tbl_w is None:
        tbl_w = OxmlElement("w:tblW")
        tbl_pr.append(tbl_w)
    tbl_w.set(qn("w:w"), str(sum(widths_dxa)))
    tbl_w.set(qn("w:type"), "dxa")
    tbl_ind = tbl_pr.first_child_found_in("w:tblInd")
    if tbl_ind is None:
        tbl_ind = OxmlElement("w:tblInd")
        tbl_pr.append(tbl_ind)
    tbl_ind.set(qn("w:w"), str(indent_dxa))
    tbl_ind.set(qn("w:type"), "dxa")
    grid = table._tbl.tblGrid
    for child in list(grid):
        grid.remove(child)
    for width in widths_dxa:
        col = OxmlElement("w:gridCol")
        col.set(qn("w:w"), str(width))
        grid.append(col)
    for row in table.rows:
        for cell, width in zip(row.cells, widths_dxa):
            set_cell_width(cell, width)
            set_cell_margins(cell)
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER


def set_plain_table_borders(table):
    """Booktabs-like table: black top/bottom rules, no box or vertical grid."""
    tbl_pr = table._tbl.tblPr
    borders = tbl_pr.first_child_found_in("w:tblBorders")
    if borders is None:
        borders = OxmlElement("w:tblBorders")
        tbl_pr.append(borders)
    for edge in ("top", "bottom"):
        elem = borders.find(qn(f"w:{edge}"))
        if elem is None:
            elem = OxmlElement(f"w:{edge}")
            borders.append(elem)
        elem.set(qn("w:val"), "single")
        elem.set(qn("w:sz"), "8")
        elem.set(qn("w:space"), "0")
        elem.set(qn("w:color"), "000000")
    for edge in ("left", "right", "insideH", "insideV"):
        elem = borders.find(qn(f"w:{edge}"))
        if elem is None:
            elem = OxmlElement(f"w:{edge}")
            borders.append(elem)
        elem.set(qn("w:val"), "nil")


def set_cell_bottom_rule(cell):
    tc_pr = cell._tc.get_or_add_tcPr()
    borders = tc_pr.first_child_found_in("w:tcBorders")
    if borders is None:
        borders = OxmlElement("w:tcBorders")
        tc_pr.append(borders)
    bottom = borders.find(qn("w:bottom"))
    if bottom is None:
        bottom = OxmlElement("w:bottom")
        borders.append(bottom)
    bottom.set(qn("w:val"), "single")
    bottom.set(qn("w:sz"), "6")
    bottom.set(qn("w:space"), "0")
    bottom.set(qn("w:color"), "000000")


def mark_row_as_header(row):
    tr_pr = row._tr.get_or_add_trPr()
    tbl_header = OxmlElement("w:tblHeader")
    tbl_header.set(qn("w:val"), "true")
    tr_pr.append(tbl_header)


def set_run_font(run, name: str, size: float, *, bold=None, italic=None, color="000000"):
    run.font.name = name
    run._element.get_or_add_rPr().rFonts.set(qn("w:ascii"), name)
    run._element.get_or_add_rPr().rFonts.set(qn("w:hAnsi"), name)
    run.font.size = Pt(size)
    run.font.color.rgb = RGBColor.from_string(color)
    if bold is not None:
        run.bold = bold
    if italic is not None:
        run.italic = italic


def add_page_field(paragraph):
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run()
    fld_char1 = OxmlElement("w:fldChar")
    fld_char1.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = " PAGE "
    fld_char2 = OxmlElement("w:fldChar")
    fld_char2.set(qn("w:fldCharType"), "end")
    run._r.extend([fld_char1, instr, fld_char2])
    set_run_font(run, "Times New Roman", 9, color="000000")


def set_core_properties(doc: Document, title: str, subject: str):
    cp = doc.core_properties
    cp.title = title
    cp.subject = subject
    cp.author = "Xihan Ma et al."
    cp.keywords = KEYWORDS
    cp.comments = "SPIE Medical Imaging 2027 submission draft prepared from the ASMUS 2026 manuscript."


def setup_abstract_styles(doc: Document):
    section = doc.sections[0]
    section.page_width = Inches(8.5)
    section.page_height = Inches(11)
    section.top_margin = Inches(1.0)
    section.bottom_margin = Inches(1.0)
    section.left_margin = Inches(1.0)
    section.right_margin = Inches(1.0)

    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal._element.rPr.rFonts.set(qn("w:ascii"), "Times New Roman")
    normal._element.rPr.rFonts.set(qn("w:hAnsi"), "Times New Roman")
    normal.font.size = Pt(10)
    normal.font.color.rgb = RGBColor.from_string("000000")
    normal.paragraph_format.space_before = Pt(0)
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.0

    for style_name, size, before, after in (
        ("Heading 1", 11, 10, 5),
        ("Heading 2", 10, 8, 4),
    ):
        style = doc.styles[style_name]
        style.font.name = "Times New Roman"
        style._element.rPr.rFonts.set(qn("w:ascii"), "Times New Roman")
        style._element.rPr.rFonts.set(qn("w:hAnsi"), "Times New Roman")
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = RGBColor.from_string("000000")
        style.paragraph_format.space_before = Pt(before)
        style.paragraph_format.space_after = Pt(after)
        style.paragraph_format.keep_with_next = True
        style.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER


def build_abstract_docx():
    assert 200 <= count_words(TECHNICAL_ABSTRACT) <= 300
    assert 50 <= count_words(PROGRAM_SUMMARY) <= 150

    doc = Document()
    set_core_properties(doc, TITLE, "SPIE MI105 abstract materials")
    setup_abstract_styles(doc)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(8)
    run = p.add_run(TITLE)
    set_run_font(run, "Times New Roman", 16, bold=True, color="000000")

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(10)
    run = p.add_run(AUTHORS)
    set_run_font(run, "Times New Roman", 12, color="000000")

    doc.add_heading("Abstract for technical review", level=1)
    p = doc.add_paragraph(TECHNICAL_ABSTRACT)
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.keep_together = True

    doc.add_heading("Summary for display in the program", level=1)
    p = doc.add_paragraph(PROGRAM_SUMMARY)
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.keep_together = True

    doc.add_heading("Keywords", level=1)
    p = doc.add_paragraph(KEYWORDS)
    p.paragraph_format.keep_together = True

    doc.save(OUT / "Abstract_v0.docx")


def setup_supplement_styles(doc: Document):
    section = doc.sections[0]
    section.page_width = Inches(8.5)
    section.page_height = Inches(11)
    section.left_margin = Inches(0.88)
    section.right_margin = Inches(0.88)
    section.top_margin = Inches(1.0)
    section.bottom_margin = Inches(1.25)
    section.header_distance = Inches(0.45)
    section.footer_distance = Inches(0.6)

    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal._element.rPr.rFonts.set(qn("w:ascii"), "Times New Roman")
    normal._element.rPr.rFonts.set(qn("w:hAnsi"), "Times New Roman")
    normal.font.size = Pt(10)
    normal.font.color.rgb = RGBColor.from_string("000000")
    normal.paragraph_format.space_before = Pt(0)
    normal.paragraph_format.space_after = Pt(4)
    normal.paragraph_format.line_spacing = 1.0

    h1 = doc.styles["Heading 1"]
    h1.font.name = "Times New Roman"
    h1._element.rPr.rFonts.set(qn("w:ascii"), "Times New Roman")
    h1._element.rPr.rFonts.set(qn("w:hAnsi"), "Times New Roman")
    h1.font.size = Pt(11)
    h1.font.bold = True
    h1.font.color.rgb = RGBColor.from_string("000000")
    h1.paragraph_format.space_before = Pt(8)
    h1.paragraph_format.space_after = Pt(5)
    h1.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    h1.paragraph_format.keep_with_next = True

    h2 = doc.styles["Heading 2"]
    h2.font.name = "Times New Roman"
    h2._element.rPr.rFonts.set(qn("w:ascii"), "Times New Roman")
    h2._element.rPr.rFonts.set(qn("w:hAnsi"), "Times New Roman")
    h2.font.size = Pt(10)
    h2.font.bold = True
    h2.font.color.rgb = RGBColor.from_string("000000")
    h2.paragraph_format.space_before = Pt(5)
    h2.paragraph_format.space_after = Pt(4)
    h2.paragraph_format.keep_with_next = True

    caption = doc.styles["Caption"]
    caption.font.name = "Times New Roman"
    caption._element.rPr.rFonts.set(qn("w:ascii"), "Times New Roman")
    caption._element.rPr.rFonts.set(qn("w:hAnsi"), "Times New Roman")
    caption.font.size = Pt(9)
    caption.font.italic = False
    caption.font.bold = False
    caption.font.color.rgb = RGBColor.from_string("000000")
    caption.paragraph_format.space_before = Pt(2)
    caption.paragraph_format.space_after = Pt(5)
    caption.paragraph_format.line_spacing = 1.0
    caption.paragraph_format.keep_together = True
    caption.paragraph_format.left_indent = Inches(0.25)
    caption.paragraph_format.right_indent = Inches(0.25)


def add_body(doc: Document, text: str, *, first_bold: str | None = None, keep=False):
    p = doc.add_paragraph()
    if first_bold and text.startswith(first_bold):
        r = p.add_run(first_bold)
        r.bold = True
        p.add_run(text[len(first_bold):])
    else:
        p.add_run(text)
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.keep_together = keep
    return p


def add_figure(doc: Document, path: Path, width: float, caption_text: str):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(0)
    p.paragraph_format.keep_with_next = True
    shape = p.add_run().add_picture(str(path), width=Inches(width))
    shape._inline.docPr.set("descr", caption_text)
    cap = doc.add_paragraph(caption_text, style="Caption")
    cap.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    return cap


def add_prompt_table(doc: Document):
    rows = [
        ("Prompt family", "Released-checkpoint setting", "Macro Dice (%)"),
        ("Box", "MedSAM - oracle box", "72.73"),
        ("", "UltraSAM - oracle box", "85.31"),
        ("Point", "SAMUS - one positive foreground point", "40.53"),
        ("", "UltraSAM - one positive foreground point", "71.21"),
        ("Text / hybrid", "MedSAM3 - generic \"object\" text", "29.17"),
        ("", "MedSAM3 - label-derived text", "39.07"),
        ("", "MedSAM3 - label-derived text + oracle box", "61.06"),
        ("", "Medical SAM3 - generic \"object\" text", "18.22"),
        ("", "Medical SAM3 - label-derived text", "26.14"),
    ]
    caption = doc.add_paragraph(
        "Table 1. Prompt-stratified descriptive results. Macro Dice is the arithmetic mean of seven "
        "dataset-level mean Dice values (BUS-BRA, GIST514, MMOTU, TNSC, UltraBones100k, UMUD, and "
        "RobLUS), giving each dataset equal weight. The table is not a cross-prompt leaderboard; confidence "
        "intervals and significance tests were not available.",
        style="Caption",
    )
    caption.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    caption.paragraph_format.keep_with_next = True
    table = doc.add_table(rows=len(rows), cols=3)
    table.alignment = WD_TABLE_ALIGNMENT.LEFT
    widths = [1510, 6685, 1510]
    set_table_geometry(table, widths, indent_dxa=110)
    set_plain_table_borders(table)
    mark_row_as_header(table.rows[0])
    for r_idx, row in enumerate(rows):
        for c_idx, value in enumerate(row):
            cell = table.rows[r_idx].cells[c_idx]
            cell.text = value
            p = cell.paragraphs[0]
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = 1.0
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if c_idx == 2 else WD_ALIGN_PARAGRAPH.LEFT
            for run in p.runs:
                set_run_font(
                    run,
                    "Times New Roman",
                    9,
                    bold=(r_idx == 0),
                    color="000000",
                )
                if r_idx == 0:
                    set_cell_bottom_rule(cell)
    spacer = doc.add_paragraph()
    spacer.paragraph_format.space_after = Pt(1)


def add_reference(doc: Document, number: int, text: str):
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Inches(0.22)
    p.paragraph_format.first_line_indent = Inches(-0.22)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.line_spacing = 1.0
    p.add_run(f"[{number}] {text}")
    for run in p.runs:
        set_run_font(run, "Times New Roman", 10)


def build_supplement_docx():
    doc = Document()
    set_core_properties(doc, TITLE, "SPIE MI105 technical review supplement")
    setup_supplement_styles(doc)

    # Page 1: title, purpose, methods, and the original ASMUS workflow figure.
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(8)
    p.paragraph_format.keep_with_next = True
    run = p.add_run(TITLE)
    set_run_font(run, "Times New Roman", 16, bold=True, color="000000")

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(10)
    p.paragraph_format.keep_with_next = True
    run = p.add_run(AUTHORS)
    set_run_font(run, "Times New Roman", 12, color="000000")

    doc.add_heading("1. PURPOSE", level=1)
    add_body(
        doc,
        "Promptable foundation models may reduce the anatomy-specific engineering and annotation burden of "
        "ultrasound segmentation, but a useful technology assessment must distinguish boundary delineation from "
        "target localization and must expose failure modes across anatomies. We assess when released promptable "
        "models provide reliable segmentation under different prompt information, target shapes, and wording "
        "specificity. The work is a model-centered technology assessment; it does not include a human-observer "
        "study or claim clinical diagnostic performance."
    )

    doc.add_heading("2. METHODS", level=1)
    add_body(
        doc,
        "Data and framework. Thirteen public 2D B-mode ultrasound datasets were harmonized through a registry-decoder "
        "pipeline into image-target samples with preserved dataset provenance. The collection contains 120,313 source "
        "images and 118,852 positive masks across 11 anatomical categories. Images remained at native resolution; "
        "model-specific conversion was performed in wrappers that returned predictions at the original resolution.",
        first_bold="Data and framework."
    )
    add_body(
        doc,
        "Models and prompts. Released checkpoints for MedSAM [1], SAMUS [2], UltraSAM [3], MedSAM3 [4], and Medical "
        "SAM3 [5] were used without fine-tuning, test-time adaptation, or dataset-specific threshold optimization. "
        "Prompt protocols comprised an oracle tight box, a label-derived clinical concept, the generic word \"object,\" "
        "and label text paired with an oracle box. For a point prompt, all foreground pixels in the complete binary mask "
        "were enumerated in row-major order and the middle element was selected; disconnected components were not "
        "handled separately. Box scaling and translation were evaluated for UltraSAM, but the source analysis did not "
        "separately record whether each perturbed box continued to contain the full target.",
        first_bold="Models and prompts."
    )
    add_body(
        doc,
        "Evaluation design. Dice was the primary summary metric; boundary distances were retained only at dataset "
        "level because physical pixel spacing was unavailable across sources. Following Metrics Reloaded [6], results "
        "are organized by prompt family, and the seven-dataset cross-model summary is an unweighted mean of dataset "
        "means. This avoids allowing the 101,343-image UltraBones100k dataset to dominate the aggregate.",
        first_bold="Evaluation design."
    )

    add_figure(
        doc,
        FIG / "Figure1_combined.png",
        6.55,
        "Figure 1. Unified benchmark pipeline from the ASMUS 2026 manuscript. Dataset-specific decoders create a shared "
        "sample schema; prompt generation and model wrappers support standardized evaluation and reporting."
    )

    # Page 2: prompt-stratified results and the plain ASMUS-style table.
    heading = doc.add_heading("3. RESULTS", level=1)
    heading.paragraph_format.page_break_before = True
    add_body(
        doc,
        "Prompt-stratified performance. Table 1 reports descriptive point estimates for the seven datasets with the "
        "cross-model results available in the source study. Within matched prompt families, box-prompted MedSAM and "
        "UltraSAM reached macro Dice values of 72.73% and 85.31%; point-prompted SAMUS and UltraSAM reached 40.53% "
        "and 71.21%; and label-prompted MedSAM3 and Medical SAM3 reached 39.07% and 26.14%. These pairs should not be "
        "combined into one model ranking because prompt interfaces and training provenance differ.",
        first_bold="Prompt-stratified performance."
    )
    add_prompt_table(doc)
    add_body(
        doc,
        "Localization and target effects. Within MedSAM3, supplementing label text with an oracle box increased macro "
        "Dice from 39.07% to 61.06%, with large gains for the pleural line (5.34% to 66.79%) and GIST514 target (23.46% "
        "to 65.53%). Even with spatial localization, bone surface/acoustic-shadow and muscle-aponeurosis targets "
        "remained difficult (30.24% and 11.35% Dice). Label specificity was most consequential in scenes with competing "
        "structures: for MedSAM3 on TNSC, replacing \"thyroid nodule\" with \"object\" reduced Dice from 72.73% to "
        "20.23%.",
        first_bold="Localization and target effects."
    )

    # Page 3: original ASMUS quantitative and qualitative figures.
    heading = doc.add_heading("4. PROMPT ROBUSTNESS AND FAILURE MODES", level=1)
    heading.paragraph_format.page_break_before = True
    add_figure(
        doc,
        FIG / "ultrasam_jitter_robustness_dice_delta.png",
        6.55,
        "Figure 2. Descriptive UltraSAM sensitivity analysis across the 13-dataset collection. (a) Box scaling and "
        "(b) translation are reported as change from each dataset's tight-box baseline; a 10% translation produced "
        "roughly a five-percentage-point mean Dice reduction, larger than comparable scaling. (c) Spearman correlations "
        "relate Dice to target shape. No uncertainty intervals or hypothesis tests were computed, and target containment "
        "after perturbation was not separately reported."
    )
    add_body(
        doc,
        "Compact, well-circumscribed lesions and masses were generally easier to delineate than line-like, elongated, "
        "or weak-boundary targets. Small box translations could redirect the model toward adjacent object-like anatomy. "
        "Text prompts added a separate recognition-and-localization burden: a generic word often sufficed for a single "
        "salient foreground target but failed when multiple plausible structures were present."
    )
    add_figure(
        doc,
        FIG / "qualitative_results_failure_cases.png",
        6.55,
        "Figure 3. Illustrative outputs spanning the seven anatomies in the cross-model reporting subset. Yellow boxes "
        "show prompts, white contours show reference masks, and red regions show predictions. Cases were selected to "
        "cover the prespecified target categories and visualize error modes supported by dataset-level trends; they "
        "are not used to estimate failure prevalence."
    )

    # Page 4: novelty, limitations, conclusion, and prior-submission disclosure.
    heading = doc.add_heading("5. NOVEL AND DISTINCTIVE ASPECTS", level=1)
    heading.paragraph_format.page_break_before = True
    add_body(
        doc,
        "The contribution is not a new architecture. Its distinct value is an ultrasound-specific consolidation of an "
        "established evaluation paradigm: 13 public datasets and 11 anatomical categories are decoded into one "
        "provenance-preserving schema; prompt families are evaluated separately; prompt imprecision and wording are "
        "treated as technology variables; and target morphology is linked to failure patterns. The resulting framework "
        "supports reproducible, task-aware evaluation of released segmentation systems and highlights cases that "
        "region-overlap averages alone can conceal."
    )

    doc.add_heading("6. LIMITATIONS", level=1)
    add_body(
        doc,
        "This analysis reports point estimates without confidence intervals or significance testing. The cross-model "
        "table contains seven of the 13 curated datasets; the other six contribute to the broader collection and "
        "UltraSAM perturbation analysis but are not claimed as complete cross-model evidence. Robustness and morphology "
        "analyses use only UltraSAM and cannot establish a universal model property. The single foreground point may be "
        "unrepresentative for disconnected or nonconvex targets, so point results are protocol-specific."
    )
    add_body(
        doc,
        "The perturbation analysis did not record whether each translated or scaled box retained the complete target; "
        "a displaced box may therefore change the localization task rather than represent boundary imprecision alone. "
        "The source study did not evaluate every model-prompt combination because released interfaces differ."
    )
    add_body(
        doc,
        "Training provenance is not uniform across releases. UltraSAM's US-43d development corpus includes public "
        "datasets that also occur in this collection (including BUS-BRA, GIST514-DB, and MMOTU-2D), and SAMUS's US30K "
        "includes BUSI and CAMUS. Results therefore characterize released-checkpoint behavior and must not be described "
        "as strict unseen-dataset zero-shot generalization or as an architecture-only comparison. The study is limited "
        "to B-mode images and does not evaluate model complexity, latency, memory, scanner/acquisition effects, or human "
        "observer performance."
    )

    doc.add_heading("7. CONCLUSION", level=1)
    add_body(
        doc,
        "Prompt type, target geometry, and dataset provenance materially change the apparent reliability of ultrasound "
        "segmentation foundation models. Practical assessments should match prompt information across systems, report "
        "target- and dataset-level behavior, and stress-test localization imprecision before integration into clinical "
        "or acquisition workflows."
    )

    doc.add_heading("8. DISCLOSURE OF PRIOR OR CONCURRENT SUBMISSIONS", level=1)
    add_body(
        doc,
        "A substantially overlapping manuscript titled \"When Do Promptable Foundation Models Simplify Ultrasound "
        "Segmentation? A Multi-Anatomy Benchmark and Prompt-Robustness Study\" was submitted to the MICCAI 2026 "
        "Workshop on Advances in Simplifying Medical Ultrasound (ASMUS), Submission 56, and was not accepted. This SPIE "
        "submission uses the same experimental results but revises the framing, prompt-stratified presentation, "
        "interpretation, and limitations; no additional validation experiments were performed. No concurrent submission "
        "is disclosed in the supplied materials."
    )

    # Page 5: references, outside the four-page technical-content count.
    heading = doc.add_heading("REFERENCES", level=1)
    heading.paragraph_format.page_break_before = True
    add_reference(doc, 1, "Ma J, He Y, Li F, et al. Segment anything in medical images. Nature Communications. 2024;15:654.")
    add_reference(doc, 2, "Lin X, Xiang Y, Yu L, Yan Z. Beyond adapting SAM: towards end-to-end ultrasound image segmentation via auto prompting. MICCAI. 2024:24-34.")
    add_reference(doc, 3, "Meyer A, Murali A, Zarin F, Mutter D, Padoy N. UltraSam: a foundation model for ultrasound using large open-access segmentation datasets. International Journal of Computer Assisted Radiology and Surgery. 2026;21:93-102.")
    add_reference(doc, 4, "Liu A, Xue R, Cao XR, et al. MedSAM3: delving into segment anything with medical concepts. arXiv:2511.19046. 2025.")
    add_reference(doc, 5, "Jiang C, Ding T, Song C, et al. Medical SAM3: a foundation model for universal prompt-driven medical image segmentation. arXiv:2601.10880. 2026.")
    add_reference(doc, 6, "Maier-Hein L, Reinke A, Godau P, et al. Metrics Reloaded: recommendations for image analysis validation. Nature Methods. 2024;21:195-212.")

    for section in doc.sections:
        footer = section.footer.paragraphs[0]
        add_page_field(footer)

    doc.save(OUT / "Supplementary_v0.docx")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    build_abstract_docx()
    build_supplement_docx()
    print(f"Technical abstract words: {count_words(TECHNICAL_ABSTRACT)}")
    print(f"Program summary words: {count_words(PROGRAM_SUMMARY)}")
    print(OUT / "Abstract_v0.docx")
    print(OUT / "Supplementary_v0.docx")
