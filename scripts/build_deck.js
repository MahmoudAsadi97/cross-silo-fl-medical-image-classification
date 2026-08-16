// Builds reports/presentation.pptx — a defense deck for the FL project.
// Run: node scripts/build_deck.js   (pptxgenjs preinstalled)
const pptxgen = require("pptxgenjs");
const path = require("path");

const REPO = path.resolve(__dirname, "..");
const FIG = (n) => path.join(REPO, "reports", "figures", n);

const INK = "12303A", PRIMARY = "0B6E7A", TEAL2 = "12A0A8", ACCENT = "E4572E",
      MUTED = "6B7B84", CARD = "EEF4F5", DARK = "0B2A31", WHITE = "FFFFFF";

const p = new pptxgen();
p.defineLayout({ name: "W", width: 13.333, height: 7.5 });
p.layout = "W";
p.author = "Mahmoud Asadi Heris";

// ---- helpers ---------------------------------------------------------------
function title(s, t, sub) {
  s.addText(t, { x: 0.55, y: 0.34, w: 12.2, h: 0.7, fontFace: "Cambria", fontSize: 29,
    bold: true, color: PRIMARY });
  if (sub) s.addText(sub, { x: 0.57, y: 1.02, w: 12.2, h: 0.4, fontFace: "Calibri",
    fontSize: 14, italic: true, color: MUTED });
}
function pic(s, name, x, y, w, h) {
  s.addImage({ path: FIG(name), x, y, w, h, sizing: { type: "contain", w, h } });
}
function bullets(s, items, x, y, w, h, sz) {
  s.addText(items.map((t, i) => ({ text: t, options: {
    bullet: { code: "2022", indent: 14 }, breakLine: true, paraSpaceAfter: 8,
    color: INK, fontSize: sz || 15 } })),
    { x, y, w, h, fontFace: "Calibri", valign: "top" });
}
function stat(s, num, label, x, y, w) {
  s.addText(num, { x, y, w, h: 0.7, fontFace: "Cambria", fontSize: 40, bold: true,
    color: ACCENT, align: "center", margin: 0 });
  s.addText(label, { x, y: y + 0.72, w, h: 0.6, fontFace: "Calibri", fontSize: 11.5,
    color: MUTED, align: "center", margin: 0 });
}
function card(s, x, y, w, h) {
  s.addShape(p.ShapeType.roundRect, { x, y, w, h, rectRadius: 0.08, fill: { color: CARD },
    line: { color: "DCE7E9", width: 1 } });
}

// ============================ 1. TITLE ======================================
let s = p.addSlide();
s.background = { color: DARK };
s.addText("Cross-Silo Federated Learning", { x: 0.8, y: 2.1, w: 11.7, h: 0.9,
  fontFace: "Cambria", fontSize: 40, bold: true, color: WHITE });
s.addText("for Privacy-Preserving Medical Image Classification", { x: 0.8, y: 3.0, w: 11.7,
  h: 0.7, fontFace: "Cambria", fontSize: 26, color: "9FD5D8" });
s.addText([
  { text: "Training a skin-lesion classifier across 6 hospitals — without ever sharing patient images",
    options: { fontSize: 15, color: "CFE6E8", italic: true, breakLine: true, paraSpaceAfter: 14 } },
  { text: "Mahmoud Asadi Heris   ·   Bachelor Creative Technologies & AI (Howest)   ·   Innovation & Research Project",
    options: { fontSize: 13, color: "8FB8BC" } },
], { x: 0.82, y: 4.0, w: 11.6, h: 1.4 });
s.addNotes("Opening: hospitals hold the data needed for good diagnostic AI, but can't share patient images. This project builds a federated system so six hospitals train one model together without sharing images, and rigorously measures the trade-offs between accuracy, privacy, security and cost.");

// ============================ 2. PROBLEM ====================================
s = p.addSlide();
title(s, "The problem & the research question",
  "Valuable medical data is locked inside institutions by law and ethics");
bullets(s, [
  "Hospitals hold exactly the data needed to train good diagnostic AI.",
  "Privacy law + ethics mean patient images usually cannot leave the building.",
  "So how do several hospitals train one strong model together — sharing no raw data?",
], 0.55, 1.6, 6.1, 2.6, 16);
card(s, 6.95, 1.55, 5.85, 3.9);
s.addText("Research question", { x: 7.2, y: 1.75, w: 5.4, h: 0.4, fontFace: "Calibri",
  bold: true, fontSize: 13, color: PRIMARY });
s.addText("How can cross-silo Federated Learning be engineered to train clinically relevant image-classification models without sharing raw data — while quantifying the trade-offs between performance, data heterogeneity, and privacy?",
  { x: 7.2, y: 2.15, w: 5.35, h: 2.1, fontFace: "Cambria", italic: true, fontSize: 16, color: INK });
s.addText("Federated Learning: each hospital trains locally and sends only model updates (numbers) to a coordinator that averages them. The images never move.",
  { x: 7.2, y: 4.3, w: 5.35, h: 1.0, fontFace: "Calibri", fontSize: 12.5, color: MUTED });
bullets(s, [
  "Benchmark: Fed-ISIC2019 (FLamby) — ~23,000 dermoscopy images, 8 diagnoses, 6 hospitals.",
  "Metric: balanced accuracy (all 8 classes count equally; random = 0.125).",
], 0.55, 4.55, 6.1, 1.6, 14);
s.addNotes("Frame the real-world stakes: this is the GDPR/clinical-AI tension. FL is the answer, and the contribution is measuring the trade-offs, not just building it. Note balanced accuracy is the strict, honest metric.");

// ============================ 3. DATA / NON-IID =============================
s = p.addSlide();
title(s, "The data is highly uneven across hospitals (non-IID)",
  "This unevenness is exactly what makes federated learning hard");
pic(s, "client_class_distribution.png", 6.5, 1.5, 6.4, 5.3);
bullets(s, [
  "6 hospitals, sizes from 7,947 down to 281 images.",
  "Some hospitals are missing whole disease classes.",
  "Each bar (right) is one hospital; colours are the 8 diagnoses — the mixes clearly differ.",
  "We quantified this: label entropy, missing-class counts, and Jensen–Shannon distance from the global mix.",
], 0.55, 1.6, 5.7, 4.0, 15.5);
s.addText("If every hospital looked the same, averaging their models would be trivial. Because they don't, naive averaging can wobble — motivating every method that follows.",
  { x: 0.55, y: 5.7, w: 5.7, h: 1.1, fontFace: "Calibri", italic: true, fontSize: 13, color: PRIMARY });
s.addNotes("Sub-question 1. Explain non-IID plainly: different patient populations and imaging devices per hospital. This is the core difficulty and the reason for FedProx/SCAFFOLD etc.");

// ============================ 4. APPROACH ===================================
s = p.addSlide();
title(s, "Approach: one fair pipeline, rigorously validated",
  "Everything held constant except the method — so differences are real");
const rows4 = [
  ["Fair-comparison protocol", "One identical training loop for every strategy; only a small 'hook' differs. Same data splits, model, and compute budget."],
  ["Reproducibility", "Config tiers (smoke/dev/full), global seeding, per-run manifest (git commit + hardware), ≥3 seeds for headline numbers."],
  ["Independent verification", "A torch-free suite of 20 checks proves the maths (aggregation, privacy accountant, robust aggregators) without a GPU."],
  ["From simulation to hardware", "Same engine runs in a fast simulation and as a real networked system on a laptop + Raspberry Pi."],
];
let yy = 1.65;
rows4.forEach(([h, b]) => {
  card(s, 0.55, yy, 12.25, 1.15);
  s.addText(h, { x: 0.8, y: yy + 0.12, w: 3.5, h: 0.9, fontFace: "Calibri", bold: true,
    fontSize: 14.5, color: PRIMARY, valign: "middle" });
  s.addText(b, { x: 4.4, y: yy + 0.1, w: 8.2, h: 0.95, fontFace: "Calibri", fontSize: 13.5,
    color: INK, valign: "middle" });
  yy += 1.32;
});
s.addNotes("This slide answers 'how do I trust your numbers'. Emphasise the structural fairness (only the method changes) and the torch-free proof suite — correctness is an acceptance test, not an afterthought.");

// ============================ 5. DOES IT WORK ===============================
s = p.addSlide();
title(s, "Does federation work? Yes — the core result",
  "FedAvg sits between training alone and pooling all data — with no data shared");
pic(s, "phase23_comparison_full.png", 5.9, 1.55, 7.0, 3.3);
stat(s, "0.32", "FedAvg (federated)", 0.55, 1.7, 1.9);
stat(s, "0.46", "Centralized (pooled)", 2.55, 1.7, 1.9);
stat(s, "0.21", "Local-only (alone)", 4.55, 1.7, 1.9);
bullets(s, [
  "FedAvg recovers ~45% of the gap between training alone and pooling everything.",
  "Bought without any raw image leaving a hospital (answers Q2).",
  "Client drift is textbook: FedAvg 8.6 ≫ FedProx 1.6 ≫ SCAFFOLD 0.9.",
], 0.55, 5.0, 12.2, 1.8, 15);
s.addNotes("Headline. The 'sandwich' (local < FL < centralized) is the value proposition demonstrated. Random is 0.125, so 0.32 is well above chance. Drift ordering matches theory (Q3, Q4).");

// ============================ 6. KEY FINDING ================================
s = p.addSlide();
s.background = { color: CARD };
title(s, "An honest finding: drift control ≠ accuracy");
s.addText("The fancier methods reduce client drift 5–9× exactly as theory predicts — but on this data that did NOT translate into higher accuracy.",
  { x: 0.55, y: 1.5, w: 12.2, h: 0.9, fontFace: "Calibri", fontSize: 17, color: INK });
const cols = [
  ["FedProx", "0.224", "proximal term"],
  ["SCAFFOLD", "0.217", "control variates"],
  ["FedAdam", "—", "adaptive server (unstable)"],
  ["FedAvg", "0.320", "the strong baseline"],
];
let cx = 0.7;
cols.forEach(([n, v, d], i) => {
  const hl = i === 3;
  s.addShape(p.ShapeType.roundRect, { x: cx, y: 2.7, w: 2.9, h: 2.5, rectRadius: 0.1,
    fill: { color: hl ? PRIMARY : WHITE }, line: { color: "DCE7E9", width: 1 } });
  s.addText(n, { x: cx, y: 2.95, w: 2.9, h: 0.5, align: "center", fontFace: "Calibri",
    bold: true, fontSize: 17, color: hl ? WHITE : INK });
  s.addText(v, { x: cx, y: 3.5, w: 2.9, h: 0.8, align: "center", fontFace: "Cambria",
    bold: true, fontSize: 34, color: hl ? WHITE : ACCENT });
  s.addText(d, { x: cx, y: 4.4, w: 2.9, h: 0.6, align: "center", fontFace: "Calibri",
    fontSize: 12, color: hl ? "CFE6E8" : MUTED });
  cx += 3.05;
});
s.addText("We report this straight: on Fed-ISIC2019, well-tuned FedAvg wins. FedProx/SCAFFOLD deliver stability, not accuracy; FedAdam was too tuning-sensitive to converge. Honesty over hype.",
  { x: 0.55, y: 5.5, w: 12.2, h: 1.2, fontFace: "Calibri", italic: true, fontSize: 14, color: PRIMARY });
s.addNotes("This negative result is a strength — it shows you tested critically rather than assuming the newest method wins. Examiners reward this. FedAdam: server_lr too high exploded; conservative regime over-diverged.");

// ============================ 7. DP =========================================
s = p.addSlide();
title(s, "Privacy 1 — a mathematical guarantee (Differential Privacy)",
  "A knob, ε, sets the strength: smaller ε = more privacy = lower accuracy");
pic(s, "dp_privacy_utility_full.png", 6.6, 1.55, 6.3, 5.1);
bullets(s, [
  "DP adds calibrated noise so no single patient noticeably changes the model.",
  "We built our OWN privacy accountant (pure Python) to verify ε — not trusting the library blindly.",
  "The curve makes the price explicit: ε≈38 keeps 0.21; very strong privacy (ε≈5) drops to 0.15.",
], 0.55, 1.6, 5.75, 3.4, 15.5);
s.addText("Turns 'we added noise' into a quantified promise a decision-maker can dial.",
  { x: 0.55, y: 5.2, w: 5.75, h: 0.8, fontFace: "Calibri", italic: true, fontSize: 14, color: PRIMARY });
s.addNotes("Sub-questions 5–6. Define ε in one line. Stress the independent accountant — that's rigour. δ=1e-5 < 1/N per client.");

// ============================ 8. MIA ========================================
s = p.addSlide();
title(s, "Privacy 2 — proving it works: a membership-inference attack",
  "We didn't just claim privacy — we attacked the model");
pic(s, "mia_auc.png", 7.0, 1.6, 5.6, 5.0);
stat(s, "0.55", "attack AUC — no privacy (leaks)", 0.7, 1.9, 2.6);
stat(s, "0.50", "attack AUC — with DP (= chance)", 3.6, 1.9, 2.6);
bullets(s, [
  "The attack guesses whether a specific patient was in the training data.",
  "0.5 means the attacker learns nothing; above 0.5 means leakage.",
  "DP measurably shuts the attack down — empirical proof, not a promise.",
], 0.55, 3.6, 6.1, 2.6, 15.5);
s.addNotes("Sub-question 7. Attack-and-defend is how security is demonstrated. The effect is modest because the model is a mild regulariser, but the direction and reduction are clear.");

// ============================ 9. SECURE AGG =================================
s = p.addSlide();
title(s, "Privacy 3 — hiding each hospital's update (Secure Aggregation)",
  "The coordinator learns the sum, never any single contribution");
pic(s, "secure_agg.png", 6.7, 1.55, 6.2, 5.1);
bullets(s, [
  "Different threat from DP: this hides each hospital's individual update from the server.",
  "Hospitals add pairwise secret masks that cancel out when summed.",
  "Verified exact: the recovered average matches plaintext to ~13 decimal places (error ≈ 3×10⁻¹³).",
], 0.55, 1.6, 5.85, 3.4, 15.5);
s.addText("Closes the 'curious server' risk: the coordinator gets the average it needs, but can't inspect any hospital.",
  { x: 0.55, y: 5.2, w: 5.85, h: 0.9, fontFace: "Calibri", italic: true, fontSize: 14, color: PRIMARY });
s.addNotes("Bonawitz 2017 pairwise masks. Emphasise it's verified numerically. Together with DP + MIA this is the full confidentiality story.");

// ============================ 10. ROBUSTNESS (bonus) =======================
s = p.addSlide();
title(s, "Security — what if a hospital is malicious? (Poisoning & defense)",
  "BONUS: the integrity side of security — attack and defend");
pic(s, "robustness.png", 6.5, 1.55, 6.4, 4.7);
bullets(s, [
  "2 of 6 hospitals send poisoned updates to sabotage the model.",
  "Plain FedAvg (a mean) has no defence — it collapses to random (0.125).",
  "Robust aggregators — median, trimmed-mean, Krum — bound each client's influence and recover (~0.18–0.19).",
], 0.55, 1.6, 5.75, 3.4, 15);
stat(s, "0.125", "FedAvg under attack (random)", 0.7, 5.2, 2.6);
stat(s, "0.19", "robust aggregation (defended)", 3.55, 5.2, 2.6);
s.addNotes("New dimension. Confidentiality (DP/MIA/secure-agg) is only half of security; this adds integrity. Krum: pick the update most consistent with the honest majority. Verified in the torch-free suite.");

// ============================ 11. COMMS (bonus) ============================
s = p.addSlide();
title(s, "Efficiency — cutting network cost 50× (communication)",
  "BONUS: FL's real bottleneck is bandwidth, acute for edge devices");
pic(s, "comms.png", 5.7, 1.7, 7.2, 3.6);
stat(s, "50×", "smaller uploads", 0.7, 1.8, 2.3);
stat(s, "≈0", "accuracy lost", 3.1, 1.8, 2.3);
bullets(s, [
  "Send only each hospital's largest 1% of update values (layer-wise top-k), skip the rest.",
  "44.7 MB → 0.89 MB per round, with accuracy essentially unchanged (0.202 vs 0.205).",
  "Pairs with the Raspberry Pi: the slowest, weakest device gains the most from sending less.",
], 0.55, 5.3, 12.2, 1.6, 14.5);
s.addNotes("New. Practicality dimension. Warm-started + layer-wise top-k so the classifier head isn't starved. Directly supports the edge/Pi story on the next slide.");

// ============================ 12. GRAD-CAM (bonus) =========================
s = p.addSlide();
title(s, "Trust — looking inside the model (Explainability)",
  "BONUS: an honest limitation, surfaced deliberately");
pic(s, "gradcam.png", 5.9, 1.7, 7.0, 3.9);
bullets(s, [
  "Grad-CAM highlights the image regions that drove each decision.",
  "On our compute-limited model, attention is diffuse and sometimes lands on image borders/artifacts, not the lesion.",
  "This is a documented 'shortcut' in dermoscopy AI — and catching it is exactly why explainability belongs in the pipeline.",
], 0.55, 1.75, 5.1, 4.0, 14.5);
s.addText("Reported openly: a production model would need stronger, artifact-free attention before clinical use.",
  { x: 0.55, y: 5.85, w: 12.2, h: 0.7, fontFace: "Calibri", italic: true, fontSize: 13.5, color: ACCENT });
s.addNotes("Own this as a limitation, not a failure. The responsible-AI point is running the check and being honest about what it revealed.");

// ============================ 13. PI (bonus) ===============================
s = p.addSlide();
s.background = { color: DARK };
s.addText("Real hardware: from simulation to a Raspberry Pi", { x: 0.7, y: 0.45, w: 12,
  h: 0.7, fontFace: "Cambria", bold: true, fontSize: 28, color: WHITE });
s.addText("BONUS: genuinely distributed FL — a laptop coordinator + a Raspberry Pi 5 as a real 'edge hospital'",
  { x: 0.72, y: 1.15, w: 12, h: 0.4, fontFace: "Calibri", italic: true, fontSize: 14, color: "9FD5D8" });
s.addImage({ path: FIG("live_straggler.png"), x: 6.7, y: 1.7, w: 6.1, h: 4.6,
  sizing: { type: "contain", w: 6.1, h: 4.6 } });
s.addText([
  { text: "Only model weights crossed the network — the Pi's images never left it.", options: { color: "E6F2F3", fontSize: 15, bullet: { code: "2022" }, breakLine: true, paraSpaceAfter: 10 } },
  { text: "It exposes the straggler problem: the Pi is ~17× slower per round, and synchronous FedAvg waits for it.", options: { color: "E6F2F3", fontSize: 15, bullet: { code: "2022" }, breakLine: true, paraSpaceAfter: 10 } },
  { text: "A real systems insight a single-machine simulation cannot show — and why the 50× compression matters.", options: { color: "E6F2F3", fontSize: 15, bullet: { code: "2022" } } },
], { x: 0.7, y: 2.0, w: 5.8, h: 3.6, fontFace: "Calibri", valign: "top" });
stat(s, "17×", "slower on the Pi", 0.9, 5.7, 2.4);
s.addText("An ≈€80 device meaningfully joins training — with no data leaving it.", { x: 3.6, y: 5.95,
  w: 3.0, h: 1.0, fontFace: "Calibri", italic: true, fontSize: 13, color: "9FD5D8", valign: "middle" });
s.addNotes("The credibility upgrade: it's not just a simulation. Describe the setup (Flower/gRPC over the LAN). The straggler is the headline systems finding.");

// ============================ 14. VALIDATION ===============================
s = p.addSlide();
title(s, "Validation, rigour & honesty",
  "Correctness treated as an acceptance requirement, not an afterthought");
const v = [
  ["20 / 20", "torch-free maths checks (aggregation, SCAFFOLD, privacy accountant, secure-agg, robust aggregators)"],
  ["≥ 3 seeds", "for every headline number, reported as mean ± std"],
  ["Manifests", "every run records git commit, tier, and hardware — fully reproducible"],
  ["Honest negatives", "FedAdam and the Grad-CAM limitation reported straight, not hidden"],
];
let vy = 1.7;
v.forEach(([n, d]) => {
  s.addText(n, { x: 0.7, y: vy, w: 2.5, h: 0.9, fontFace: "Cambria", bold: true, fontSize: 26,
    color: PRIMARY, valign: "middle" });
  s.addText(d, { x: 3.4, y: vy, w: 9.3, h: 0.9, fontFace: "Calibri", fontSize: 15, color: INK,
    valign: "middle" });
  vy += 1.25;
});
s.addNotes("Pre-empt the 'how do I trust this' question. The torch-free suite is the standout — it proves the science independent of the GPU stack.");

// ============================ 15. CONCLUSION ===============================
s = p.addSlide();
s.background = { color: DARK };
s.addText("Conclusion — when is FL preferable?", { x: 0.7, y: 0.5, w: 12, h: 0.8,
  fontFace: "Cambria", bold: true, fontSize: 30, color: WHITE });
s.addText([
  { text: "When data cannot leave an institution, FL beats both training alone and slow, risky data-sharing agreements — it recovered ~45% of the centralized-vs-isolated gap with no raw data leaving a hospital.", options: { color: "E6F2F3", fontSize: 15.5, bullet: { code: "2022" }, breakLine: true, paraSpaceAfter: 12 } },
  { text: "Privacy is layered and verified: differential privacy (with a self-built accountant), a defeated membership attack, and secure aggregation.", options: { color: "E6F2F3", fontSize: 15.5, bullet: { code: "2022" }, breakLine: true, paraSpaceAfter: 12 } },
  { text: "Security covers both sides: confidentiality and integrity (robust aggregation defeats a poisoning attack).", options: { color: "E6F2F3", fontSize: 15.5, bullet: { code: "2022" }, breakLine: true, paraSpaceAfter: 12 } },
  { text: "Practical & real: 50× cheaper communication, demonstrated live on a Raspberry Pi.", options: { color: "E6F2F3", fontSize: 15.5, bullet: { code: "2022" }, breakLine: true, paraSpaceAfter: 12 } },
  { text: "Future work: EfficientNet backbone for higher accuracy, client-level DP, full-tier robustness runs, artifact-free explainability.", options: { color: "9FD5D8", fontSize: 14, italic: true, bullet: { code: "2022" } } },
], { x: 0.75, y: 1.5, w: 11.9, h: 5.2, fontFace: "Calibri", valign: "top" });
s.addNotes("Answer Q8 directly, then recap the four pillars (works / private / secure / practical) and end on credible future work.");

// ============================ 16. THANK YOU ================================
s = p.addSlide();
s.background = { color: PRIMARY };
s.addText("Thank you", { x: 0.8, y: 2.6, w: 11.7, h: 1.0, fontFace: "Cambria", bold: true,
  fontSize: 46, color: WHITE });
s.addText("Questions welcome — dashboard, technical report, and code are ready to explore.",
  { x: 0.82, y: 3.7, w: 11.5, h: 0.6, fontFace: "Calibri", fontSize: 16, color: "CFEBEC" });
s.addText("github.com/MahmoudAsadi97/cross-silo-fl-medical-image-classification", { x: 0.82,
  y: 4.4, w: 11.5, h: 0.4, fontFace: "Consolas", fontSize: 12, color: "AEDFE1" });
s.addNotes("Offer the live dashboard walkthrough and, if possible, the Pi demo. Have docs/design.md and speaker notes handy.");

p.writeFile({ fileName: path.join(REPO, "reports", "presentation.pptx") })
  .then((f) => console.log("wrote", f));
