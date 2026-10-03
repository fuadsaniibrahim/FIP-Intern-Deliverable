"""Pull the evidence files out of the executed notebook's saved outputs.

Nothing is recomputed here: every CSV and PNG is copied from what the notebook
actually printed or plotted. Run from the module folder after executing the notebook:

    python evidence/extract_evidence.py
"""
import base64
import csv
from html.parser import HTMLParser
from pathlib import Path

import nbformat

HERE = Path(__file__).resolve().parent
NOTEBOOK = HERE.parent / "Module1_Ensemble_and_XAI.ipynb"


class TableParser(HTMLParser):
    """Collect the rows of the first <table> in a pandas HTML repr."""

    def __init__(self):
        super().__init__()
        self.rows, self.row, self.cell, self.in_cell = [], None, "", False

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self.row = []
        elif tag in ("td", "th"):
            self.in_cell, self.cell = True, ""

    def handle_endtag(self, tag):
        if tag in ("td", "th"):
            self.row.append(self.cell.strip())
            self.in_cell = False
        elif tag == "tr":
            self.rows.append(self.row)

    def handle_data(self, data):
        if self.in_cell:
            self.cell += data


def find_cell(nb, marker):
    matches = [c for c in nb.cells if c.cell_type == "code" and marker in c.source]
    assert len(matches) == 1, f"expected one cell containing {marker!r}, found {len(matches)}"
    cell = matches[0]
    assert not any(o.output_type == "error" for o in cell.outputs), f"cell with {marker!r} has an error"
    return cell


def images(cell):
    return [o.data["image/png"] for o in cell.outputs if "image/png" in o.get("data", {})]


def tables(cell):
    return [o.data["text/html"] for o in cell.outputs if "text/html" in o.get("data", {})]


def save_table(html, name, index_label="model"):
    parser = TableParser()
    parser.feed(html)
    header, *body = [r for r in parser.rows if r]
    header = [index_label] + header[1:]
    with open(HERE / name, "w", newline="") as f:
        csv.writer(f).writerows([header] + body)
    print("wrote", name, f"({len(body)} rows)")


def save_png(b64, name):
    (HERE / name).write_bytes(base64.b64decode(b64))
    print("wrote", name)


nb = nbformat.read(NOTEBOOK, as_version=4)

save_table(tables(find_cell(nb, "val_table = "))[0], "results_validation.csv")
save_table(tables(find_cell(nb, "cross_val_score(m, X_train"))[0], "cv_results.csv")

final_cell = find_cell(nb, "test_table = ")
save_table(tables(final_cell)[0], "results_test.csv")
save_png(images(final_cell)[0], "fig_confusion_test.png")

save_png(images(find_cell(nb, "RocCurveDisplay.from_estimator"))[0], "fig_roc_validation.png")

beeswarm, bar = images(find_cell(nb, "shap.plots.beeswarm"))
save_png(beeswarm, "fig_shap_beeswarm.png")
save_png(bar, "fig_shap_bar.png")

# second waterfall in that cell is the case closest to the decision boundary
save_png(images(find_cell(nb, "shap.plots.waterfall"))[1], "fig_shap_waterfall_uncertain.png")
save_png(images(find_cell(nb, "LimeTabularExplainer("))[0], "fig_lime_uncertain.png")
