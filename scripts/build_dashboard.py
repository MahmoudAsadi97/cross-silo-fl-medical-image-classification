#!/usr/bin/env python3
"""Build a single self-contained dashboard.html a non-expert can read top-to-bottom.

Every result figure is embedded as base64 (portable single file, no external assets), and
each is wrapped with a plain-language "what this is", a "what we added & why" note, and a
"why it matters" note, plus an at-a-glance results strip and a glossary. Torch-free.

    python scripts/build_dashboard.py    ->    reports/dashboard.html
"""
from __future__ import annotations

import base64
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FIG = REPO / "reports" / "figures"
OUT = REPO / "reports" / "dashboard.html"


def fig(name: str, caption: str = "") -> str:
    p = FIG / name
    if not p.exists():
        return f'<div class="missing">[figure not yet generated: {name}]</div>'
    b64 = base64.b64encode(p.read_bytes()).decode()
    cap = f"<figcaption>{caption}</figcaption>" if caption else ""
    return (f'<figure><img alt="{name}" src="data:image/png;base64,{b64}">{cap}</figure>')


def callout(kind: str, title: str, body: str) -> str:
    return f'<div class="callout {kind}"><span class="ct">{title}</span><div>{body}</div></div>'


def details(summary: str, body: str) -> str:
    return f"<details><summary>{summary}</summary><div class='det'>{body}</div></details>"


CSS = """
:root{--ink:#1a2233;--muted:#5a6577;--line:#e5e9f0;--accent:#1f6fb2;--bg:#f6f8fb;
--add:#0b7285;--addbg:#e7f5f8;--why:#2b8a3e;--whybg:#ebfbee;--warn:#a6772a;--warnbg:#fff8ea;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
font:16px/1.6 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif}
a{color:var(--accent);text-decoration:none}
header.hero{background:linear-gradient(135deg,#1f6fb2,#14507f);color:#fff;padding:44px 22px 30px}
header.hero .wrap{max-width:960px;margin:0 auto}
header.hero h1{margin:0 0 6px;font-size:30px;line-height:1.2}
header.hero p.sub{margin:0;opacity:.92;font-size:17px;max-width:760px}
header.hero p.by{margin:14px 0 0;opacity:.8;font-size:13px}
nav.toc{position:sticky;top:0;z-index:9;background:#fff;border-bottom:1px solid var(--line);
padding:10px 22px;overflow-x:auto;white-space:nowrap}
nav.toc a{font-size:13px;color:var(--muted);margin-right:14px}
nav.toc a:hover{color:var(--accent)}
main{max-width:960px;margin:0 auto;padding:8px 22px 60px}
.glance{display:flex;flex-wrap:wrap;gap:12px;margin:22px 0}
.glance .kpi{flex:1 1 150px;background:#fff;border:1px solid var(--line);border-radius:12px;
padding:14px 16px}
.glance .kpi b{display:block;font-size:22px;color:var(--accent)}
.glance .kpi span{font-size:12.5px;color:var(--muted)}
section.card{background:#fff;border:1px solid var(--line);border-radius:16px;
padding:24px 26px;margin:22px 0;box-shadow:0 1px 2px rgba(20,40,80,.04)}
section.card h2{margin:0 0 4px;font-size:22px}
section.card .tag{display:inline-block;font-size:11px;font-weight:700;letter-spacing:.04em;
text-transform:uppercase;color:#fff;background:var(--accent);border-radius:20px;
padding:3px 10px;margin-bottom:10px}
section.card .tag.new{background:#12805c}
p.lead{font-size:16.5px;margin:.3em 0 1em}
figure{margin:16px 0;text-align:center}
figure img{max-width:100%;border:1px solid var(--line);border-radius:10px;background:#fff}
figcaption{font-size:12.5px;color:var(--muted);margin-top:6px}
.callout{border-radius:10px;padding:12px 14px;margin:12px 0;font-size:14.5px}
.callout .ct{display:block;font-weight:700;font-size:12px;text-transform:uppercase;
letter-spacing:.03em;margin-bottom:3px}
.callout.add{background:var(--addbg);color:var(--add)}
.callout.add .ct{color:var(--add)}
.callout.why{background:var(--whybg);color:var(--why)}
.callout.why .ct{color:var(--why)}
.callout.warn{background:var(--warnbg);color:var(--warn)}
.callout.warn .ct{color:var(--warn)}
.callout div{color:var(--ink)}
table.res{border-collapse:collapse;width:100%;margin:12px 0;font-size:14px}
table.res th,table.res td{border:1px solid var(--line);padding:7px 10px;text-align:left}
table.res th{background:#f0f4f9}
table.res td.n{text-align:right;font-variant-numeric:tabular-nums}
.best{background:#eafaf1;font-weight:600}
.bad{background:#fdeceb}
details{margin:10px 0;border:1px solid var(--line);border-radius:8px;padding:0 12px;background:#fbfcfe}
details summary{cursor:pointer;font-size:13.5px;font-weight:600;color:var(--accent);padding:9px 0}
details .det{font-size:14px;color:var(--muted);padding:0 0 10px}
.gloss dt{font-weight:700;margin-top:10px}
.gloss dd{margin:2px 0 0;color:var(--muted);font-size:14.5px}
footer{max-width:960px;margin:0 auto;padding:20px 22px 50px;color:var(--muted);font-size:13px}
.missing{color:#a33;font-style:italic;padding:14px;border:1px dashed #e2b0ad;border-radius:8px}
"""

GLANCE = [
    ("0.32", "FedAvg balanced accuracy (vs 0.46 pooled, 0.21 alone)"),
    ("0", "raw patient images ever shared"),
    ("8&times;", "attack drop resisted by robust aggregation"),
    ("50&times;", "smaller uploads at ~same accuracy"),
    ("17&times;", "slower on the Raspberry Pi (straggler)"),
    ("0.50", "membership-attack AUC after privacy (= random)"),
]


def section(num, tag, title, lead, body, new=False):
    tagcls = "tag new" if new else "tag"
    return (f'<section class="card" id="s{num}"><span class="{tagcls}">{tag}</span>'
            f"<h2>{title}</h2><p class='lead'>{lead}</p>{body}</section>")


def build() -> str:
    parts = []

    # ---- 1. problem -----------------------------------------------------------
    parts.append(section(
        1, "The problem", "Why hospitals can&rsquo;t just pool their data",
        "Hospitals hold exactly the data needed to train good diagnostic AI &mdash; but "
        "privacy law and ethics mean patient images usually cannot leave the building. So how "
        "do several hospitals train one strong model together <b>without sharing any raw images</b>?",
        callout("add", "Our answer in one line",
                "<b>Federated Learning (FL):</b> each hospital trains on its own images and "
                "sends only the <i>model updates</i> (numbers) to a coordinator, which averages "
                "them into a shared model. The images never move.")
        + callout("why", "Why it matters",
                  "This is the whole premise of the project: collaboration on sensitive medical "
                  "data with privacy built in, on a real benchmark (Fed-ISIC2019 &mdash; ~23,000 "
                  "skin-lesion photos, 8 diagnoses, 6 hospitals).")))

    # ---- 2. heterogeneity -----------------------------------------------------
    parts.append(section(
        2, "The data", "The six hospitals are very different (&ldquo;non-IID&rdquo;)",
        "Before training, we measured how different the hospitals&rsquo; data is. They are "
        "<b>highly uneven</b>: sizes range from 7,947 down to 281 images, and some hospitals are "
        "missing whole disease classes. This unevenness is what makes federated learning hard.",
        fig("client_class_distribution.png",
            "Each bar is one hospital; colours are the 8 diagnoses. The mixes are clearly different.")
        + callout("why", "Why it matters",
                  "If every hospital had the same mix, averaging their models would be easy. Because "
                  "they don&rsquo;t, naive averaging can wobble &mdash; which motivates every method below.")
        + details("How we measured it (for the curious)",
                  "We computed each hospital&rsquo;s class distribution and its distance from the "
                  "global mix (Jensen&ndash;Shannon divergence), plus counts of missing classes and "
                  "label entropy. Hospital&nbsp;1 is the most skewed (misses 2 classes).")))

    # ---- 3. comparison --------------------------------------------------------
    comp_table = """
    <table class="res"><tr><th>Method</th><th>Balanced accuracy</th><th>Client drift</th><th>Plain meaning</th></tr>
    <tr class="best"><td>Centralized (all data pooled)</td><td class="n">0.456</td><td class="n">&ndash;</td><td>best case, but needs sharing</td></tr>
    <tr><td><b>FedAvg</b> (federated)</td><td class="n">0.320</td><td class="n">8.56</td><td>our main FL model &mdash; no sharing</td></tr>
    <tr><td>FedProx</td><td class="n">0.224</td><td class="n">1.55</td><td>controls drift, doesn&rsquo;t add accuracy</td></tr>
    <tr><td>SCAFFOLD</td><td class="n">0.217</td><td class="n">0.92</td><td>controls drift most, slower</td></tr>
    <tr><td>Local-only (each alone)</td><td class="n">0.209</td><td class="n">&ndash;</td><td>no collaboration = worst</td></tr>
    </table>"""
    parts.append(section(
        3, "Core result", "Does federation actually work? Yes.",
        "We compared training strategies fairly (same model, data, and budget &mdash; only the "
        "method changes). <b>FedAvg reaches 0.32</b>, sitting between training alone (0.21) and the "
        "ideal of pooling all data (0.46). It recovers ~45% of that gap <b>without sharing images</b>.",
        comp_table + fig("phase23_comparison_full.png",
            "Left: accuracy by method (higher is better). Right: &ldquo;client drift&rdquo; &mdash; how far hospitals pull apart.")
        + callout("add", "An honest finding we&rsquo;re proud of",
                  "The fancier methods (FedProx, SCAFFOLD, and FedAdam which we also built) <b>cut drift "
                  "but did not beat plain FedAvg</b> on this data. We report that straight: on this "
                  "benchmark, well-tuned FedAvg is the strong baseline. Honesty &gt; hype.")
        + callout("why", "Why 0.32 and not higher?",
                  "&ldquo;Balanced accuracy&rdquo; is a strict metric on 8 imbalanced classes (random = 0.125). "
                  "We also used a smaller model at lower resolution so it fits a laptop GPU and works with "
                  "privacy. The <i>point</i> was measuring the trade-offs, not chasing a leaderboard.")))

    # ---- 4. DP ----------------------------------------------------------------
    parts.append(section(
        4, "Privacy &mdash; 1", "Adding a mathematical privacy guarantee (Differential Privacy)",
        "Model updates can still leak information. <b>Differential Privacy (DP)</b> adds calibrated "
        "noise so no single patient changes the result much. A knob, &epsilon; (epsilon), sets the "
        "strength: <b>smaller &epsilon; = more privacy = lower accuracy</b>.",
        fig("dp_privacy_utility_full.png",
            "The privacy&ndash;accuracy trade-off: as privacy tightens (left), accuracy falls, smoothly and predictably.")
        + callout("add", "What we added & why",
                  "We built our <i>own</i> privacy accountant from scratch (pure Python) to double-check the "
                  "&epsilon; numbers, rather than trusting the library blindly &mdash; so the guarantee is verified.")
        + callout("why", "Why it matters",
                  "It turns &ldquo;we added noise&rdquo; into a <b>quantified promise</b>: e.g. &epsilon;&asymp;38 keeps "
                  "accuracy at 0.21; very strong privacy (&epsilon;&asymp;5) drops it to 0.15. Leaders can pick the point.")))

    # ---- 5. MIA ---------------------------------------------------------------
    parts.append(section(
        5, "Privacy &mdash; 2", "Proving the privacy works: a membership-inference attack",
        "We didn&rsquo;t just claim privacy &mdash; we <b>attacked</b> the model. A membership-inference "
        "attack tries to tell whether a specific patient was in the training set. Score 0.5 = the "
        "attacker is guessing; above 0.5 = leakage.",
        fig("mia_auc.png", "Without privacy the attack scores 0.55 (leaks); with DP it drops to 0.50 (pure chance).")
        + callout("why", "Why it matters",
                  "This is empirical proof, not a promise: DP <b>measurably shuts down</b> the attack "
                  "(0.55&rarr;0.50). Attack-and-defend is exactly how security is demonstrated.")))

    # ---- 6. secure agg --------------------------------------------------------
    parts.append(section(
        6, "Privacy &mdash; 3", "Hiding each hospital&rsquo;s update from the coordinator (Secure Aggregation)",
        "Even the coordinator shouldn&rsquo;t see any one hospital&rsquo;s update. With <b>secure "
        "aggregation</b>, hospitals add secret masks that cancel out when summed &mdash; the server "
        "learns only the <i>total</i>, never an individual contribution.",
        fig("secure_agg.png",
            "Each hospital&rsquo;s update is scrambled (tall bars) yet the final sum is exact (flat line at zero error).")
        + callout("why", "Why it matters",
                  "The coordinator gets the average it needs to improve the model, but cannot inspect any single "
                  "hospital &mdash; closing the &ldquo;curious server&rdquo; risk. Verified exact to 13 decimal places.")))

    # ---- 7. robustness (NEW) --------------------------------------------------
    rob_table = """
    <table class="res"><tr><th>Aggregation method (under attack)</th><th>Balanced accuracy</th><th>Verdict</th></tr>
    <tr><td>Clean FedAvg (no attack, reference)</td><td class="n">0.173</td><td>baseline</td></tr>
    <tr class="bad"><td>FedAvg &mdash; <b>under attack</b></td><td class="n">0.125</td><td>collapses to random</td></tr>
    <tr class="best"><td>Coordinate-median (robust)</td><td class="n">0.191</td><td>defended</td></tr>
    <tr class="best"><td>Krum (robust)</td><td class="n">0.179</td><td>defended</td></tr>
    <tr class="best"><td>Trimmed-mean (robust)</td><td class="n">0.172</td><td>defended</td></tr>
    </table>"""
    parts.append(section(
        7, "Security &mdash; integrity", "What if a hospital is malicious? (Poisoning &amp; defense)",
        "The privacy work protects <i>confidentiality</i>. But a <b>malicious or hacked hospital</b> could "
        "send poisoned updates to sabotage the shared model. We simulated exactly that (2 of 6 hospitals "
        "attacking) and tested defenses.",
        rob_table + fig("robustness.png",
            "Red = plain FedAvg under attack, stuck at the random floor. Green/blue/purple = robust methods, which recover.")
        + callout("add", "What we added & why", "This is a brand-new dimension: a <b>model-poisoning attack</b> plus "
                  "three <b>robust aggregators</b> (median, trimmed-mean, Krum) that limit how much any one hospital can "
                  "move the model. It completes the security story: confidentiality <i>and</i> integrity.")
        + callout("why", "Why it matters",
                  "A single defenseless average is wrecked by 2 bad actors (drops to 0.125 = random guessing); robust "
                  "aggregation shrugs the attack off and keeps full accuracy. Essential for real hospital consortia.")))

    # ---- 8. comms (NEW) -------------------------------------------------------
    comms_table = """
    <table class="res"><tr><th>Update sent</th><th>Upload size / round</th><th>Compression</th><th>Balanced accuracy</th></tr>
    <tr><td>Full model (100%)</td><td class="n">44.7 MB</td><td class="n">1&times;</td><td class="n">0.205</td></tr>
    <tr><td>Top 10% of values</td><td class="n">8.9 MB</td><td class="n">5&times;</td><td class="n">0.197</td></tr>
    <tr class="best"><td>Top 1% of values</td><td class="n">0.89 MB</td><td class="n">50&times;</td><td class="n">0.202</td></tr>
    </table>"""
    parts.append(section(
        8, "Efficiency", "Cutting the network cost 50&times; (communication)",
        "Federated learning&rsquo;s real-world bottleneck is bandwidth &mdash; every round each hospital "
        "uploads the whole model (~45&nbsp;MB). We send only the <b>largest 1% of update values</b> and "
        "skip the rest.",
        comms_table + fig("comms.png",
            "Accuracy stays flat (~0.20) across the whole range &mdash; even shrinking uploads 50&times;.")
        + callout("add", "What we added & why",
                  "Top-k <b>sparsification</b>: send only the values that matter. New here, and it pairs perfectly "
                  "with the edge/Raspberry-Pi story below (a weak device on a slow link).")
        + callout("why", "Why it matters",
                  "50&times; smaller uploads for essentially the same accuracy (0.202 vs 0.205). Makes FL practical for "
                  "bandwidth-limited or edge hospitals.")))

    # ---- 9. explainability ----------------------------------------------------
    parts.append(section(
        9, "Trust", "Looking inside the model (Explainability)",
        "For a clinical model, we should check <b>where it looks</b>. Grad-CAM highlights the image "
        "regions that drove each decision.",
        fig("gradcam.png", "Grad-CAM overlays on test lesions using the federated model.")
        + callout("warn", "An honest limitation we surfaced",
                  "On our small, compute-limited model the attention is <b>diffuse and sometimes lands on image "
                  "borders/artifacts rather than the lesion</b> &mdash; a known &ldquo;shortcut&rdquo; in dermoscopy AI. "
                  "We report this openly: it&rsquo;s exactly the kind of problem explainability is meant to catch, and "
                  "it would gate a real deployment.")
        + callout("why", "Why it matters",
                  "Running the check &mdash; and being honest about what it revealed &mdash; is the responsible-AI point. "
                  "A production model would need stronger, artifact-free attention before clinical use.")))

    # ---- 10. Pi ---------------------------------------------------------------
    parts.append(section(
        10, "Real hardware", "From simulation to a real device: the Raspberry Pi",
        "Everything above is a <i>simulation</i> on one computer. To prove it&rsquo;s genuinely "
        "distributed, we ran the system across <b>two real machines</b>: a laptop as the coordinator "
        "and a <b>Raspberry Pi 5 as a real &lsquo;edge hospital&rsquo;</b>, talking over the network.",
        fig("live_accuracy.png", "The global model, trained live across the laptop + Pi, holds ~0.20 accuracy.")
        + fig("live_straggler.png", "Time per round: the Pi (orange) is ~17&times; slower than the laptop GPU (blue).")
        + callout("add", "What we added & why",
                  "A real client&ndash;server deployment (Flower/gRPC): the Pi trains on its own data and sends only "
                  "weights. It turns &ldquo;federated&rdquo; from a claim into a demonstrated fact on hardware.")
        + callout("why", "Why it matters",
                  "It shows an ~&euro;80 device can join training with no data leaving it &mdash; and it exposes the "
                  "<b>straggler problem</b> (the whole round waits for the slowest device), a real systems insight the "
                  "simulation can&rsquo;t show. This is why the 50&times; compression above matters for edge devices.")))

    # ---- glossary -------------------------------------------------------------
    gloss = [
        ("Federated Learning (FL)", "Training one shared model across many data owners by exchanging model updates, not raw data."),
        ("Cross-silo", "The FL setting with a few, persistent, data-rich clients &mdash; here, hospitals."),
        ("Non-IID", "The clients&rsquo; data is not identically distributed &mdash; each hospital has a different mix. This makes FL harder."),
        ("Balanced accuracy", "Average accuracy across all 8 classes (so rare diseases count equally). Random guessing = 0.125."),
        ("Client drift", "How far the hospitals&rsquo; locally-trained models pull apart each round; large drift can destabilise the average."),
        ("Differential Privacy (DP) / &epsilon;", "A math guarantee that no single patient noticeably affects the model; &epsilon; is the privacy strength (smaller = more private)."),
        ("Membership inference", "An attack that guesses whether a specific person was in the training data. AUC 0.5 = attacker learns nothing."),
        ("Secure aggregation", "A scheme where the server can add up hospital updates but cannot read any single one."),
        ("Poisoning / Byzantine", "A malicious client sending bad updates to sabotage the model."),
        ("Robust aggregation (Krum, median, trimmed-mean)", "Ways to combine updates that ignore extreme/outlier (malicious) ones."),
        ("Sparsification (top-k)", "Sending only the largest few percent of update values to save bandwidth."),
        ("Straggler", "The slowest device in a synchronous round; everyone waits for it."),
        ("Grad-CAM", "A method that highlights which image regions drove a model&rsquo;s decision."),
    ]
    gl = "".join(f"<dt>{t}</dt><dd>{d}</dd>" for t, d in gloss)
    parts.append(section(11, "Reference", "Glossary &mdash; every term in plain words",
                         "Quick definitions so nothing on this page needs outside explanation.",
                         f'<dl class="gloss">{gl}</dl>'))

    nav = "".join(f'<a href="#s{i}">{t}</a>' for i, t in [
        (1, "Problem"), (2, "The data"), (3, "Does it work?"), (4, "Privacy: DP"),
        (5, "Privacy: attack"), (6, "Privacy: secure"), (7, "Security: poisoning"),
        (8, "Efficiency"), (9, "Trust"), (10, "Raspberry Pi"), (11, "Glossary")])

    glance = "".join(f'<div class="kpi"><b>{v}</b><span>{k}</span></div>' for v, k in GLANCE)

    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Cross-Silo Federated Learning &mdash; Project Dashboard</title><style>{CSS}</style></head>
<body>
<header class="hero"><div class="wrap">
<h1>Cross-Silo Federated Learning for Privacy-Preserving Medical Imaging</h1>
<p class="sub">Six hospitals train one skin-lesion classifier together &mdash; <b>without ever sharing
patient images</b> &mdash; and we measure the trade-offs between accuracy, privacy, security, and cost.
This page walks through the whole project in plain language; read it top to bottom.</p>
<p class="by">Mahmoud Asadi Heris &middot; Bachelor Creative Technologies &amp; AI (Howest) &middot; Innovation &amp; Research Project</p>
</div></header>
<nav class="toc">{nav}</nav>
<main>
<div class="glance">{glance}</div>
{''.join(parts)}
</main>
<footer>Every number on this page comes from a script in this project and a figure it generated
(<code>reports/figures/</code>). Green boxes = why it matters; teal boxes = what we added and why;
amber boxes = an honest limitation. Built by <code>scripts/build_dashboard.py</code>.</footer>
</body></html>"""


if __name__ == "__main__":
    OUT.write_text(build(), encoding="utf-8")
    kb = OUT.stat().st_size / 1024
    print(f"wrote {OUT}  ({kb:.0f} KB)")
