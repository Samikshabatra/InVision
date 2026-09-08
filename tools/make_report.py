"""Generate the project report PDF.

A single self-contained document explaining what InVision does, how it works,
why each design decision was taken, and where its limits are - written to be
read before a viva and defended in one.
"""

from __future__ import annotations

import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate, Frame, KeepTogether, ListFlowable, ListItem, NextPageTemplate,
    PageBreak, PageTemplate, Paragraph, Preformatted, Spacer, Table, TableStyle,
)

OUT = Path(__file__).resolve().parents[1] / "InVision_Project_Report.pdf"

INK = colors.HexColor("#16202b")
MUTED = colors.HexColor("#5c6773")
ACCENT = colors.HexColor("#1f5fbf")
RULE = colors.HexColor("#d7dde4")
SOFT = colors.HexColor("#eef3fa")
WARN = colors.HexColor("#8a5a00")
WARNSOFT = colors.HexColor("#fdf3e2")

base = getSampleStyleSheet()

S = {
    "title": ParagraphStyle("title", parent=base["Title"], fontName="Helvetica-Bold",
                            fontSize=27, leading=32, textColor=INK, spaceAfter=6),
    "subtitle": ParagraphStyle("subtitle", parent=base["Normal"], fontName="Helvetica",
                               fontSize=13, leading=18, textColor=MUTED, spaceAfter=20),
    "h1": ParagraphStyle("h1", parent=base["Heading1"], fontName="Helvetica-Bold",
                         fontSize=17, leading=21, textColor=INK,
                         spaceBefore=16, spaceAfter=8),
    "h2": ParagraphStyle("h2", parent=base["Heading2"], fontName="Helvetica-Bold",
                         fontSize=12.5, leading=16, textColor=ACCENT,
                         spaceBefore=12, spaceAfter=5),
    "body": ParagraphStyle("body", parent=base["Normal"], fontName="Helvetica",
                           fontSize=10, leading=14.6, textColor=INK,
                           alignment=TA_JUSTIFY, spaceAfter=7),
    "bullet": ParagraphStyle("bullet", parent=base["Normal"], fontName="Helvetica",
                             fontSize=10, leading=14.2, textColor=INK, spaceAfter=3),
    "code": ParagraphStyle("code", parent=base["Code"], fontName="Courier",
                           fontSize=8.2, leading=10.6, textColor=INK),
    "cap": ParagraphStyle("cap", parent=base["Normal"], fontName="Helvetica-Oblique",
                          fontSize=8.6, leading=11.5, textColor=MUTED, spaceAfter=10),
    "qa_q": ParagraphStyle("qa_q", parent=base["Normal"], fontName="Helvetica-Bold",
                           fontSize=10.2, leading=14, textColor=ACCENT, spaceAfter=3),
    "qa_a": ParagraphStyle("qa_a", parent=base["Normal"], fontName="Helvetica",
                           fontSize=10, leading=14.4, textColor=INK,
                           alignment=TA_JUSTIFY, spaceAfter=11),
}


def para(text, style="body"):
    return Paragraph(text, S[style])


def bullets(items, style="bullet"):
    return ListFlowable(
        [ListItem(Paragraph(t, S[style]), leftIndent=12) for t in items],
        bulletType="bullet", bulletFontSize=7, bulletOffsetY=1,
        leftIndent=14, spaceAfter=8,
    )


def callout(title, text, tone="info"):
    fill = SOFT if tone == "info" else WARNSOFT
    edge = ACCENT if tone == "info" else WARN
    inner = [
        Paragraph(f"<b>{title}</b>", ParagraphStyle(
            "ct", parent=S["body"], textColor=edge, spaceAfter=3, fontSize=10)),
        Paragraph(text, ParagraphStyle("cb", parent=S["body"], spaceAfter=0)),
    ]
    t = Table([[inner]], colWidths=[165 * mm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), fill),
        ("LINEBEFORE", (0, 0), (0, -1), 2.2, edge),
        ("LEFTPADDING", (0, 0), (-1, -1), 9),
        ("RIGHTPADDING", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    return KeepTogether([t, Spacer(1, 10)])


def table(rows, widths, header=True, align_right=()):
    t = Table(rows, colWidths=widths, hAlign="LEFT", repeatRows=1 if header else 0)
    style = [
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("TEXTCOLOR", (0, 0), (-1, -1), INK),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("LINEBELOW", (0, 0), (-1, -2), 0.4, RULE),
        ("LINEBELOW", (0, -1), (-1, -1), 0.8, RULE),
    ]
    if header:
        style += [
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("BACKGROUND", (0, 0), (-1, 0), SOFT),
            ("TEXTCOLOR", (0, 0), (-1, 0), INK),
            ("LINEBELOW", (0, 0), (-1, 0), 0.9, ACCENT),
        ]
    for col in align_right:
        style.append(("ALIGN", (col, 0), (col, -1), "RIGHT"))
    t.setStyle(TableStyle(style))
    return KeepTogether([t, Spacer(1, 10)])


def qa(question, answer):
    return KeepTogether([Paragraph(question, S["qa_q"]), Paragraph(answer, S["qa_a"])])


def decorate(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(MUTED)
    canvas.drawString(22 * mm, 12 * mm, "InVision - Vision-Based Indoor Navigation")
    canvas.drawRightString(188 * mm, 12 * mm, f"Page {canvas.getPageNumber()}")
    canvas.setStrokeColor(RULE)
    canvas.setLineWidth(0.5)
    canvas.line(22 * mm, 15.5 * mm, 188 * mm, 15.5 * mm)
    canvas.restoreState()


def cover(canvas, doc):
    canvas.saveState()
    canvas.setFillColor(ACCENT)
    canvas.rect(0, 262 * mm, 210 * mm, 35 * mm, stroke=0, fill=1)
    canvas.restoreState()


def build():
    doc = BaseDocTemplate(
        str(OUT), pagesize=A4,
        leftMargin=22 * mm, rightMargin=22 * mm,
        topMargin=20 * mm, bottomMargin=20 * mm,
        title="InVision - Vision-Based Indoor Navigation",
        author="Computer Vision Trimester Project",
    )
    frame = Frame(doc.leftMargin, doc.bottomMargin,
                  doc.width, doc.height, id="main")
    doc.addPageTemplates([
        PageTemplate(id="cover", frames=[frame], onPage=cover),
        PageTemplate(id="body", frames=[frame], onPage=decorate),
    ])
    doc.build(story())
    return OUT


# ---------------------------------------------------------------- content

def story():
    s = []

    # ------------------------------------------------------------ cover
    s += [
        Spacer(1, 46 * mm),
        para("InVision", "title"),
        para("Vision-Based Indoor Navigation for Shopping Malls", "subtitle"),
        para(
            "A system that identifies where a shopper is standing from a single "
            "photograph of a nearby storefront, then guides them to any shop in the "
            "mall with an animated floor map and spoken turn-by-turn directions.",
            "body"),
        Spacer(1, 8),
        table([
            ["Site surveyed", "Nexus Koramangala - 5 floors"],
            ["Directory units mapped", "104"],
            ["Units with reference photographs", "83"],
            ["Photographs collected", "289, across 10 corridor walks"],
            ["Core method", "Visual place recognition (image retrieval) + scene-text fusion"],
            ["Deliverable", "Web application: photo in, animated route out"],
        ], [52 * mm, 113 * mm], header=False),
        Spacer(1, 10),
        para(
            "This document explains the problem, every significant design decision "
            "and why it was taken, the results with their honest limits, and a set "
            "of anticipated examiner questions with prepared answers.", "cap"),
        NextPageTemplate("body"),
        PageBreak(),
    ]

    # ------------------------------------------------------------ 1
    s += [
        para("1. The problem", "h1"),
        para(
            "A large shopping mall has many outlets spread over several floors. A "
            "visitor who wants a specific shop typically does not know which floor "
            "it is on, let alone which end of which corridor. Existing options are "
            "poor: static directory boards are only useful if you are standing at "
            "one, and outdoor mapping services do not work indoors because GPS is "
            "unreliable under a roof and no public routing API exposes indoor paths.",
            "body"),
        para(
            "The insight this project exploits is that a shopper always has one "
            "reliable signal available: <b>what they can see</b>. Point a phone at "
            "the shop in front of you and that image, by itself, says where you are.",
            "body"),
        para("What the system does", "h2"),
        bullets([
            "The user photographs any nearby storefront.",
            "The system identifies that shop, and therefore the user's position.",
            "The user types or selects a destination.",
            "The system draws the route on a floor map and reads the directions aloud.",
        ]),
        callout(
            "Scope note",
            "Localisation is to shop-level granularity, not metres. For wayfinding "
            "inside a mall that is sufficient - a shopper needs to know which shop "
            "they are beside and which way to walk, not their coordinates."),
    ]

    # ------------------------------------------------------------ 2
    s += [
        para("2. The central design decision: retrieval, not classification", "h1"),
        para(
            "The obvious first instinct is to treat this as image classification: "
            "one class per shop, train a CNN. That approach was rejected, and being "
            "able to explain why is the most important single point in this project.",
            "body"),
        table([
            ["", "Classification", "Retrieval (chosen)"],
            ["Data needed", "Hundreds of images per class", "A handful per place"],
            ["Adding a new shop", "Retrain the whole model", "Append its vectors"],
            ["Behaviour with 3 images/class", "Severe overfitting", "Works as designed"],
            ["What it learns", "A fixed decision boundary", "A general similarity space"],
        ], [40 * mm, 62 * mm, 63 * mm], align_right=()),
        para(
            "With roughly three photographs per shop, a classifier has far too few "
            "samples per class to generalise; it memorises. Retrieval sidesteps the "
            "problem entirely. A pretrained encoder maps any image into a vector "
            "space where similar scenes land close together, so recognition becomes "
            "a nearest-neighbour lookup against a gallery of reference vectors. No "
            "training is performed at all.",
            "body"),
        para(
            "This framing is standard in the literature under the name <b>Visual "
            "Place Recognition</b>, which matters for the write-up: the problem has "
            "known methods, benchmarks and failure modes rather than being invented "
            "for this project.",
            "body"),
        callout(
            "The one-line answer",
            "\"We do not classify shops, we retrieve them. With three images per "
            "shop a classifier would overfit, and every new shop would mean "
            "retraining. Retrieval needs no training and scales by appending "
            "vectors.\""),
    ]

    # ------------------------------------------------------------ 3
    s += [
        PageBreak(),
        para("3. System architecture", "h1"),
        Preformatted(
            "   PHOTOGRAPH\n"
            "        |\n"
            "        +-----------------------------+\n"
            "        |                             |\n"
            "        v                             v\n"
            "  CLIP  ENCODER                 OCR  (RapidOCR)\n"
            "  512-d embedding               signage text + height\n"
            "        |                             |\n"
            "        v                             v\n"
            "  cosine similarity            match against official\n"
            "  vs gallery vectors           brand names + aliases\n"
            "        |                             |\n"
            "        +-------------+---------------+\n"
            "                      v\n"
            "               FUSED  SCORE  ->  directory unit  (WHERE AM I)\n"
            "                      |\n"
            "                      v\n"
            "         NAVIGATION GRAPH  (built from the floor maps)\n"
            "         nodes: shop units, walkways, gates, escalators\n"
            "                      |\n"
            "                      v  Dijkstra shortest path\n"
            "                      |\n"
            "        +-------------+---------------+\n"
            "        v                             v\n"
            "  ANIMATED FLOOR MAP           SPOKEN DIRECTIONS\n"
            "  (SVG, route drawn)           (Web Speech API)",
            S["code"]),
        para(
            "Two independent evidence channels feed one decision, and the result "
            "drives a graph search. The channels are deliberately different in kind "
            "- appearance and text - so they fail on different images.", "cap"),
        para("Six subsystems", "h2"),
        table([
            ["Subsystem", "Responsibility", "Key file"],
            ["Ingest", "Read capture folders into ordered frames", "src/ingest.py"],
            ["Encoders", "CLIP embeddings and OCR with text height", "src/encoders.py"],
            ["Directory", "Official floor maps as machine-readable data", "data/mall_directory.json"],
            ["Frame alignment", "Assign each photo to a directory unit", "src/frame_align.py"],
            ["Localiser", "Photo -> place, via fusion", "src/localizer.py"],
            ["Navigation", "Graph, routing, instructions", "src/mall_graph.py"],
        ], [30 * mm, 82 * mm, 53 * mm]),
    ]

    # ------------------------------------------------------------ 4
    s += [
        para("4. The data problem, and how it was solved", "h1"),
        para(
            "The survey produced 289 photographs in ten folders, one per floor and "
            "side. Critically, <b>none of them carried a label</b>. The filenames "
            "were only capture timestamps; nothing recorded which shop any photo "
            "showed. A supervised pipeline needs labels, so they had to be recovered.",
            "body"),
        para("Signals available", "h2"),
        bullets([
            "<b>Folder</b> - which floor and which side of the corridor.",
            "<b>Timestamp</b> - the order the corridor was walked, which is spatial order.",
            "<b>Signage</b> - the shop name is physically written on the shopfront.",
            "<b>The official floor maps</b> - the true list of shops and their order.",
        ]),
        para(
            "The final method aligns photographs directly onto directory units. The "
            "map states exactly which units exist on a row and in what sequence, and "
            "the walk visited them in that sequence, so the task becomes an ordered "
            "matching problem rather than free-form labelling.",
            "body"),
        para("Anchor and fill", "h2"),
        bullets([
            "A frame whose signage clearly names one unit becomes an <b>anchor</b>.",
            "Anchors must appear in row order; the best-supported non-decreasing "
            "subsequence is kept, which discards reflections and directory boards "
            "naming distant shops.",
            "Every remaining frame joins its nearest surviving anchor.",
        ]),
        callout(
            "Why not simply cluster by visual similarity",
            "That was tried first and failed instructively. Neighbouring shopfronts "
            "look alike, so the clustering merged them - a food court collapsed four "
            "stalls into one, and the merged group could only carry one name, so the "
            "other three shops disappeared from the dataset entirely. Segmenting and "
            "naming are the same problem and had to be solved together.",
            tone="warn"),
        para("Text height as the disambiguator", "h2"),
        para(
            "OCR returns every word on a facade, including promotional copy, and "
            "sale wording repeats across every frame of a shop just as reliably as "
            "the brand name does. Frequency therefore cannot separate them. What "
            "does separate them is physical size: a shopfront name is set far larger "
            "than the offer text beside it. Every OCR detection is scored by its cap "
            "height relative to the tallest text in the same photograph.",
            "body"),
        para(
            "This one change moved dozens of shops from wrong to right - SALDI to "
            "METRO, SKINCARE to SEPHORA, BUYGET to VAN HEUSEN, ORDER to KFC.", "cap"),
    ]

    # ------------------------------------------------------------ 5
    s += [
        PageBreak(),
        para("5. Recognition: fusing appearance and text", "h1"),
        para("Channel one - CLIP embeddings", "h2"),
        para(
            "Every image is encoded with CLIP ViT-B/32 into a 512-dimensional "
            "L2-normalised vector, so cosine similarity is a dot product. A query is "
            "scored against each unit by its best-matching reference image. CLIP is "
            "well suited here because it was trained on image-text pairs from the "
            "web and has therefore seen a great many brand logos and shopfronts.",
            "body"),
        para("Channel two - scene text", "h2"),
        para(
            "The query is read with OCR and the words are matched against each "
            "unit's <b>official name from the directory</b>, with an alias table for "
            "fascias that abbreviate (M&amp;S for Marks and Spencer) and fuzzy matching "
            "for the characteristic OCR failure of swallowing a leading letter "
            "(ESTSIDE for WESTSIDE, DASICS for ASICS).",
            "body"),
        callout(
            "A subtle bug worth describing in the viva",
            "Originally the text channel matched against words harvested from a "
            "unit's own gallery images. That is self-defeating: if a group is "
            "labelled ALDO but actually holds photographs of the shop next door, its "
            "vocabulary fills with the neighbour's brand, and a genuine photo of that "
            "neighbour then scores highest against the wrong shop. The mislabelling "
            "was teaching itself. Matching against the directory name instead breaks "
            "the feedback loop, because the directory is independent of how the "
            "photographs were grouped.",
            tone="warn"),
        para("Why fusion helps", "h2"),
        para(
            "The channels fail differently. Two outlets of one brand look alike and "
            "read alike, so neither channel alone separates them - but they sit at "
            "different positions on different floors. A shopfront photographed at a "
            "steep angle may read poorly yet still match visually. Conversely a shop "
            "with no reference photographs cannot be matched visually at all, but its "
            "sign is perfectly readable.",
            "body"),
        para(
            "Measured contribution of each channel, on the held-out split, under the "
            "labelling in force at the time of the ablation:", "body"),
        table([
            ["Configuration", "Top-1", "Top-3", "Top-5", "Floor"],
            ["Visual only (CLIP)", "78.0%", "89.8%", "93.2%", "88.1%"],
            ["Text only (OCR)", "74.6%", "93.2%", "96.6%", "94.9%"],
            ["Fusion 0.7 / 0.3", "93.3%", "98.3%", "98.3%", "96.7%"],
        ], [55 * mm, 27 * mm, 27 * mm, 27 * mm, 27 * mm],
            align_right=(1, 2, 3, 4)),
        para(
            "Fusion beats the better single channel by more than 15 points. That "
            "gap, not the absolute number, is the defensible finding: the two "
            "signals are genuinely complementary rather than redundant.", "cap"),
        para("Signage override", "h2"),
        para(
            "A late refinement, added after real misrecognitions were reported. If "
            "the query's own signage names one unit clearly and beats the runner-up "
            "by a margin, that decides the answer outright. The reasoning is simple: "
            "if the sign says ALDO, it is Aldo, regardless of which reference image "
            "happens to look closest. This also lets the system name shops that were "
            "never photographed, which pure retrieval can never do. Where two "
            "fascias are both legible - common in wide survey shots - the result is "
            "reported as a close call and the rival is offered as a one-tap "
            "correction rather than silently guessed.",
            "body"),
    ]

    # ------------------------------------------------------------ 6
    s += [
        PageBreak(),
        para("6. Navigation", "h1"),
        para(
            "The navigation graph is built from the <b>official floor directory</b>, "
            "not from the photographs. This distinction is deliberate and worth "
            "stating plainly: the survey missed some shops and mislabelled others, "
            "and a system restricted to what the camera captured could never route "
            "to a shop the survey got wrong. Photographs supply the ability to "
            "recognise; the directory supplies the world.",
            "body"),
        para("Floor model", "h2"),
        Preformatted(
            "  front gate (west)                              back gate (east)\n"
            "      |                                                  |\n"
            "      |   +--------------- TOP ROW -----------------+    |\n"
            "      +---|                                         |----+\n"
            "      |   |   [esc west]     (atrium)   [esc east]  |    |\n"
            "      +---|                                         |----+\n"
            "          +------------- BOTTOM ROW -----------------+",
            S["code"]),
        para(
            "Each floor is a rectangular walkway around a central atrium, with shops "
            "along two rows. Shoppers cross between rows at four points: the gates at "
            "each end and the two escalator banks either side of the atrium.", "cap"),
        para(
            "An earlier version of this model was wrong, and correcting it is a good "
            "illustration of why ground-truth data matters. Before the floor maps "
            "were available the layout had been inferred from the capture order, "
            "which suggested a single loop with one escalator at its origin. The maps "
            "showed <b>four escalators in two banks</b> in the middle of the floor. "
            "With the corrected model the router chooses whichever bank is nearer, "
            "which the earlier model could not do.",
            "body"),
        para("Routing", "h2"),
        bullets([
            "Nodes: shop units, walkway points, gates, escalator banks.",
            "Edges weighted by distance, modelled from average shopfront frontage.",
            "Vertical edges connect each escalator bank across adjacent floors.",
            "Dijkstra returns the shortest path; escalator cost is set high enough "
            "that staying on one floor is preferred when the walk is comparable.",
        ]),
        para(
            "The node path is then translated into human instructions: runs along a "
            "row become \"walk toward the back gate for about 70m, past X and Y\", a "
            "change of row becomes \"cross at the east escalator\", and a change of "
            "floor becomes a single multi-level escalator instruction rather than one "
            "per level.",
            "body"),
        callout(
            "Honest framing of distances",
            "The graph is topological, not surveyed. Distances come from average "
            "shopfront frontage rather than measurement, so step order and direction "
            "are reliable while the metre figures are estimates. Pacing out one "
            "corridor would calibrate them."),
    ]

    # ------------------------------------------------------------ 7
    s += [
        para("7. The application", "h1"),
        table([
            ["Layer", "Technology", "Notes"],
            ["Backend", "FastAPI + Uvicorn", "Model loaded once at startup"],
            ["Vision", "CLIP ViT-B/32 via transformers", "CPU, ~0.15 s per image"],
            ["OCR", "RapidOCR (ONNX runtime)", "No PaddlePaddle dependency"],
            ["Graph", "NetworkX", "Dijkstra over the floor graph"],
            ["Frontend", "Single self-contained HTML page", "No build step, no CDN"],
            ["Speech", "Web Speech API", "Runs offline, no network call"],
        ], [26 * mm, 58 * mm, 81 * mm]),
        para("Interface", "h2"),
        bullets([
            "Photo input using the rear camera on a phone.",
            "Recognition result with a confidence figure, and one-tap alternatives "
            "when the decision is close.",
            "Searchable destination list covering every unit on all five floors.",
            "Animated floor plan that mirrors the printed directory - shopfront "
            "boxes in two rows, atrium, gates, both escalator banks - with the route "
            "drawn from the origin shop to the destination shop.",
            "Turn-by-turn list, read aloud on demand with the step being spoken "
            "highlighted, and an auto-play preference that is remembered.",
        ]),
        para("Endpoints", "h2"),
        table([
            ["GET /api/stores", "All directory units, grouped by floor"],
            ["GET /api/layout", "Floor geometry for the map renderer"],
            ["POST /api/locate", "Photo upload, returns ranked places"],
            ["POST /api/route", "Origin + destination, returns steps and map polylines"],
            ["GET /review", "Label verification tool"],
        ], [42 * mm, 123 * mm], header=False),
    ]

    # ------------------------------------------------------------ 8
    s += [
        PageBreak(),
        para("8. Results, and what they do and do not show", "h1"),
        para(
            "The strongest result is the ablation in Section 5: fusing appearance "
            "with scene text beats either channel alone by a wide margin. That "
            "finding is robust because it is a comparison under identical conditions.",
            "body"),
        para(
            "The <b>absolute</b> accuracy figures deserve more care, and being "
            "candid about this is likely to earn more credit than quoting a single "
            "flattering number.",
            "body"),
        para("Limitation one - the evaluation split is optimistic", "h2"),
        para(
            "Held-out query images were captured seconds apart from their own "
            "gallery images, at nearly the same angle and lighting. Matching them is "
            "close to trivial, so the figures are an upper bound rather than a "
            "realistic estimate. A fair evaluation needs photographs taken on a "
            "different day, from different positions.",
            "body"),
        para("Limitation two - the ground truth is itself inferred", "h2"),
        para(
            "The labels used as ground truth were produced by the alignment pipeline, "
            "not by a human. When the localiser was improved so that a photograph of "
            "Tommy Hilfiger is correctly named Tommy Hilfiger, the benchmark scored "
            "that as an <b>error</b>, because the stored label for that image still "
            "said Calvin Klein. The measured number fell even though the behaviour "
            "improved.",
            "body"),
        callout(
            "How to present this",
            "\"Our benchmark measures agreement with automatically generated labels, "
            "and we can demonstrate cases where the system is right and the label is "
            "wrong. Rather than quote a number we cannot defend, we built a review "
            "tool so the labels can be verified by hand - that is the work that makes "
            "the metric meaningful.\"",
            tone="warn"),
        para("The review tool", "h2"),
        para(
            "A browser tool that shows all 289 photographs in walk order, each beside "
            "its assigned unit and the expected order from the floor map, with "
            "keyboard navigation and a bulk \"shift the rest\" action for the common "
            "off-by-one drift. Corrections are stored and treated as ground truth by "
            "the build pipeline, overriding anything inferred. Two corrections during "
            "testing recovered a shop the pipeline had lost entirely.",
            "body"),
        para("Verified qualitative fixes", "h2"),
        table([
            ["Query photograph", "Before", "After"],
            ["Aldo storefront", "Forest Essentials (wrong)", "Aldo"],
            ["Tommy Hilfiger storefront", "Calvin Klein (wrong)", "Tommy Hilfiger"],
        ], [55 * mm, 55 * mm, 55 * mm]),
        para(
            "Both were adjacent-shop confusions - the signature of an off-by-one "
            "labelling error. Tommy Hilfiger now resolves correctly despite having "
            "no reference photographs at all, because its sign is legible.", "cap"),
    ]

    # ------------------------------------------------------------ 9
    s += [
        para("9. Known limitations", "h1"),
        bullets([
            "<b>21 of 104 units have no reference photographs.</b> They can be "
            "routed to, and named if their sign is readable, but cannot be matched "
            "visually. One further walk closes this gap.",
            "<b>Wide shots capture two fascias.</b> Where both names are legible the "
            "system reports a close call rather than guessing; a shopper standing in "
            "front of one shop rarely produces such framing.",
            "<b>Distances are modelled, not measured.</b> Order and direction are "
            "reliable; metres are estimates.",
            "<b>Two upper floors are thinly covered.</b> Thirteen photographs across "
            "the cinema and management floors is enough to navigate to but not enough "
            "to recognise reliably.",
            "<b>Single-mall model.</b> The directory schema generalises, but the "
            "geometry assumes a two-row floor around a central atrium.",
        ]),
        para("Future work", "h2"),
        bullets([
            "Complete the label review pass, then re-measure on a fresh test set.",
            "Extract frames from the walkthrough videos to expand the gallery well "
            "beyond three views per shop.",
            "Estimate heading from the query image so instructions can say left and "
            "right rather than referring to the gates.",
            "Adopt the IMDF schema for the map data so it generalises to other malls.",
        ]),
    ]

    # ------------------------------------------------------------ 10
    s += [
        PageBreak(),
        para("10. Anticipated examiner questions", "h1"),
        para(
            "Prepared answers to the questions most likely to be asked. The value is "
            "in the reasoning, not the wording - the point is to be able to justify "
            "each decision.", "cap"),
        qa("Why not train a CNN classifier?",
           "Because we have around three images per shop. A classifier needs many "
           "samples per class and would memorise rather than generalise, and every "
           "new shop would require retraining. Retrieval needs no training: we embed "
           "the reference images once and recognition is a nearest-neighbour lookup. "
           "Adding a shop means appending vectors."),
        qa("Why CLIP rather than ResNet or a fine-tuned backbone?",
           "CLIP is trained on image-text pairs from the web, so its embedding space "
           "already separates brands and shopfront styles without any fine-tuning. "
           "A shopfront is largely defined by its logo and signage, which is exactly "
           "the kind of content CLIP has seen. It also runs adequately on CPU."),
        qa("Why add OCR at all if CLIP already works?",
           "Because the two channels fail on different images, which the ablation "
           "shows: fusion beats the better single channel by more than 15 points. "
           "Text also reaches shops that have no reference photographs, which "
           "retrieval fundamentally cannot."),
        qa("How do you weight the two channels, and how was that chosen?",
           "0.7 visual to 0.3 text, set as the default before evaluating rather than "
           "tuned afterwards on the test set. We report the full sweep so the "
           "sensitivity is visible."),
        qa("Your accuracy dropped after an improvement. Explain that.",
           "The benchmark grades against automatically generated labels. When we "
           "fixed the localiser so a Tommy Hilfiger photograph is named Tommy "
           "Hilfiger, the stored label still said Calvin Klein, so a correct answer "
           "was counted wrong. We can demonstrate this on specific images. That is "
           "why we built the label review tool: the metric only becomes meaningful "
           "once the labels are verified."),
        qa("How do you know your labels are wrong?",
           "By searching the OCR output for brands the pipeline reported as never "
           "photographed. Fourteen of fifteen were present in the signage, which "
           "proves the photographs exist and the loss happened in grouping. That "
           "diagnostic is what drove the rewrite from clustering to directory "
           "alignment."),
        qa("Why is the graph built from the map rather than from your data?",
           "Because the map is authoritative and our survey is not. Building from "
           "the photographs meant any shop we missed simply did not exist in the "
           "app. Building from the directory gives 104 routable destinations across "
           "five floors instead of 75 across three."),
        qa("How does multi-floor routing work?",
           "Each floor is a graph of walkway nodes; the two escalator banks appear on "
           "every floor and are connected vertically between adjacent floors. "
           "Dijkstra over the combined graph therefore picks the nearer bank "
           "automatically. Escalator traversal carries a cost high enough that the "
           "route prefers staying on one floor when the walk is comparable."),
        qa("What happens if recognition is wrong?",
           "Every downstream direction would be wrong, so the system reports "
           "confidence rather than hiding it. When the top two candidates are close "
           "it says so and offers the alternatives as one-tap corrections. The "
           "signage override also means a legible sign outranks a merely similar-"
           "looking reference image."),
        qa("Is this deployable as it stands?",
           "As a demonstrator, yes - it runs end to end on a laptop and works from a "
           "phone on the same network. For production it would need the label review "
           "completed, a fresh evaluation set, the missing shops photographed, and "
           "the distances calibrated against real measurements."),
    ]

    # ------------------------------------------------------------ 11
    s += [
        PageBreak(),
        para("11. Running the system", "h1"),
        Preformatted(
            "# one-time data preparation (results are cached)\n"
            "python tools/extract_zips.py     # unpack the capture folders\n"
            "python tools/embed_all.py        # CLIP embeddings   (~50 s, 289 images)\n"
            "python tools/ocr_all.py          # OCR + text height (~8 min)\n"
            "python build_dataset.py          # align photos to directory units\n"
            "\n"
            "# use\n"
            "python api.py                    # web app at 127.0.0.1:8000\n"
            "python evaluate.py               # ablation table\n"
            "python predict.py photo.jpg      # identify one photograph\n"
            "python route.py --from-photo shot.jpg --to DECATHLON\n"
            "\n"
            "# label verification\n"
            "python api.py  ->  open /review",
            S["code"]),
        para("Repository map", "h2"),
        table([
            ["data/mall_directory.json", "The floor maps as structured data - the source of truth"],
            ["data/label_review.json", "Human label corrections, override everything inferred"],
            ["data/derived/", "Embeddings, indices, OCR cache, store list, reports"],
            ["src/frame_align.py", "Photographs to directory units (anchor and fill)"],
            ["src/localizer.py", "Fusion, signage override, ranking"],
            ["src/mall_graph.py", "Floor geometry, routing, instruction generation"],
            ["web/index.html", "The application, including map and speech"],
            ["web/review.html", "Label verification tool"],
        ], [46 * mm, 119 * mm], header=False),
        para("Talking points, in priority order", "h2"),
        bullets([
            "Retrieval over classification, and why the data forces that choice.",
            "Fusion of appearance and scene text, with the ablation as evidence.",
            "Text height as the signal that separates a brand from its sale poster.",
            "The unlabelled-data problem, and recovering labels by ordered alignment "
            "against the official map.",
            "Building the graph from the authoritative map rather than from our own "
            "incomplete survey.",
            "Honest evaluation: what the numbers show, what they cannot, and the "
            "tool built to fix that.",
        ]),
        Spacer(1, 6),
        callout(
            "If you remember one thing",
            "This project is Visual Place Recognition, not logo classification. Every "
            "significant decision follows from that framing and from the constraint "
            "of very few images per place."),
    ]

    return s


if __name__ == "__main__":
    path = build()
    print(f"wrote {path}")
    sys.exit(0)
