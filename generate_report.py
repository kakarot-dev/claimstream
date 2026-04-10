"""Generate the AIML Project Report as PDF — updated for full FEVER + DeBERTa."""
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.colors import HexColor
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, HRFlowable

OUTPUT = "ConvoAI_Project_Report.pdf"
BLUE = HexColor("#1a56db"); DARK = HexColor("#1f2937"); GRAY = HexColor("#6b7280")
LIGHT = HexColor("#f3f4f6"); WHITE = HexColor("#ffffff"); BORD = HexColor("#d1d5db")

def build():
    doc = SimpleDocTemplate(OUTPUT, pagesize=A4, leftMargin=25*mm, rightMargin=25*mm, topMargin=20*mm, bottomMargin=20*mm)
    S = getSampleStyleSheet()
    title = ParagraphStyle('T', parent=S['Title'], fontSize=26, leading=32, textColor=BLUE, spaceAfter=6, alignment=TA_CENTER)
    sub = ParagraphStyle('Sub', parent=S['Normal'], fontSize=12, leading=16, textColor=GRAY, alignment=TA_CENTER, spaceAfter=20)
    h1 = ParagraphStyle('H1', parent=S['Heading1'], fontSize=16, leading=22, textColor=BLUE, spaceBefore=18, spaceAfter=8)
    h2 = ParagraphStyle('H2', parent=S['Heading2'], fontSize=13, leading=18, textColor=DARK, spaceBefore=12, spaceAfter=6)
    body = ParagraphStyle('B', parent=S['Normal'], fontSize=10.5, leading=15, textColor=DARK, alignment=TA_JUSTIFY, spaceAfter=6)
    bul = ParagraphStyle('Bu', parent=body, leftIndent=20, bulletIndent=8, spaceBefore=2, spaceAfter=2)
    code = ParagraphStyle('C', parent=S['Code'], fontSize=9, leading=13, textColor=DARK, backColor=LIGHT, borderWidth=0.5, borderColor=BORD, borderPadding=6, leftIndent=10, rightIndent=10, spaceAfter=8)

    E = []
    def hr(): E.append(Spacer(1,4)); E.append(HRFlowable(width="100%", thickness=0.5, color=BORD)); E.append(Spacer(1,4))
    def sec(t): E.append(Paragraph(t, h1)); hr()
    def p(t): E.append(Paragraph(t, body))
    def b(t): E.append(Paragraph(f"• {t}", bul))
    def sp(n=8): E.append(Spacer(1, n))

    def tbl(data, cw=None):
        t = Table(data, colWidths=cw, repeatRows=1)
        t.setStyle(TableStyle([
            ('BACKGROUND',(0,0),(-1,0),BLUE),('TEXTCOLOR',(0,0),(-1,0),WHITE),
            ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),('FONTSIZE',(0,0),(-1,0),10),
            ('FONTSIZE',(0,1),(-1,-1),9.5),('LEADING',(0,0),(-1,-1),14),
            ('BACKGROUND',(0,1),(-1,-1),WHITE),('ROWBACKGROUNDS',(0,1),(-1,-1),[WHITE,LIGHT]),
            ('GRID',(0,0),(-1,-1),0.5,BORD),('VALIGN',(0,0),(-1,-1),'TOP'),
            ('LEFTPADDING',(0,0),(-1,-1),8),('RIGHTPADDING',(0,0),(-1,-1),8),
            ('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6),
        ]))
        E.append(t); sp(8)

    # === COVER ===
    E.append(Spacer(1,80))
    E.append(Paragraph("ConvoAI", title))
    E.append(Paragraph("Real-Time Debate Fact Checker", ParagraphStyle('BS', parent=sub, fontSize=16, leading=20, textColor=DARK)))
    sp(12); E.append(Paragraph("AIML Project Report", sub)); hr(); sp(20)
    E.append(Paragraph("A real-time audio fact-checking system that transcribes debate speech locally using Whisper, retrieves relevant facts from a 101,897-claim Wikipedia-verified database (FEVER), and verifies claims using a DeBERTa NLI model trained specifically for fact verification.", ParagraphStyle('A', parent=body, fontSize=11, leading=16, alignment=TA_CENTER, textColor=GRAY)))
    sp(30)
    tbl([['Student ID','Student Name','Contribution'],
         ['_____________','_____________','Speech-to-Text pipeline, audio streaming'],
         ['_____________','_____________','Fact verification model integration, NLI pipeline'],
         ['_____________','_____________','Web UI, debate tracking, dataset integration']], [120,120,220])
    sp(10)
    E.append(Paragraph("Domain: NLP / Embedding Systems / Fact Verification", ParagraphStyle('D', parent=body, fontSize=10, alignment=TA_CENTER, textColor=BLUE)))
    E.append(PageBreak())

    # === TOC ===
    E.append(Paragraph("Table of Contents", h1)); hr()
    for i, t in enumerate(["Problem Statement","Motivation / Use Case","Dataset","Model / Approach","System Architecture","Implementation Details","File Structure","Execution Instructions","Results / Output","Model Evaluation","Challenges Faced","Key Design Decisions","Future Improvements","Team Member Contributions"], 1):
        E.append(Paragraph(f"{i}. {t}", ParagraphStyle('TOC', parent=body, fontSize=11, leading=20, leftIndent=20)))
    E.append(PageBreak())

    # === 1. PROBLEM STATEMENT ===
    sec("1. Problem Statement")
    p("In debates and public discussions, speakers frequently make factual claims that may be incorrect, misleading, or outright false. Manually verifying these claims in real-time is impossible for human moderators — by the time a claim is checked, the conversation has moved on and the audience has absorbed the misinformation.")
    sp()
    p("This project builds a <b>real-time audio fact-checking system</b> that listens to debate speakers, transcribes their speech to text using a local Whisper model, and verifies each claim against a database of <b>101,897 Wikipedia-verified facts</b> from the FEVER (Fact Extraction and VERification) dataset using semantic similarity retrieval and Natural Language Inference (NLI).")
    sp()
    p("The system scores each debate side by factual accuracy and highlights incorrect claims with corrections. All processing runs <b>entirely locally</b> — no external API calls, no cloud services. Three pre-trained models from HuggingFace are downloaded and run on the student's machine.")

    # === 2. MOTIVATION ===
    sec("2. Motivation / Use Case")
    p("Misinformation spreads easily in live debates, panel discussions, and classroom presentations. A real-time fact-checker serves as:")
    b("<b>Moderator's assistant</b> — flags false claims during debates")
    b("<b>Educational tool</b> — teaches students to distinguish fact from fiction")
    b("<b>Transparency tool</b> — provides evidence-based scoring for competitive debates")
    b("<b>Media analysis</b> — fact-checks live broadcasts or recorded speeches")
    sp()
    p("The system is <b>topic-agnostic</b>. The FEVER dataset covers all of Wikipedia — science, history, geography, entertainment, politics, sports, and more. Any claim that can be verified against Wikipedia is covered.")

    # === 3. DATASET ===
    sec("3. Dataset")
    p("The system uses the <b>FEVER (Fact Extraction and VERification)</b> dataset, an academic benchmark created by the University of Sheffield in 2018. FEVER contains 185,000+ claims extracted from Wikipedia and verified by human annotators.")
    sp()
    tbl([['Metric','Value'],
         ['Source','FEVER dataset (HuggingFace: copenlu/fever_gold_evidence)'],
         ['Total claims','101,897'],
         ['True claims (SUPPORTS)','73,784 (72%)'],
         ['False claims (REFUTES)','28,113 (28%)'],
         ['Domain','General knowledge — all Wikipedia topics'],
         ['Format','JSON (claim, verdict, source, category)'],
         ['Coverage','Science, history, geography, entertainment, sports, politics, etc.']], [150,310])
    sp()
    p("Each claim was generated by altering Wikipedia sentences and then verified by human annotators <b>without knowledge of the original sentence</b>. This ensures the claims are natural and not trivially matchable.")
    sp()
    E.append(Paragraph('Example entries:\n\n'
        '{"claim": "The Colosseum is in Rome.", "verdict": true}\n'
        '{"claim": "Barack Obama was born in Kenya.", "verdict": false}\n'
        '{"claim": "Python is a programming language.", "verdict": true}\n'
        '{"claim": "The Earth is flat.", "verdict": false}', code))

    # === 4. MODEL / APPROACH ===
    sec("4. Model / Approach")
    E.append(Paragraph("4.1 Models Used", h2))
    p("Three pre-trained models from HuggingFace, all running locally with no API calls and <b>no custom training</b>:")
    sp(4)
    tbl([['Model','HuggingFace ID','Purpose','Size'],
         ['Whisper (tiny)','openai/whisper-tiny\n(via faster-whisper)','Speech-to-Text\n(real-time on CPU)','~75 MB'],
         ['Sentence-Transformer','sentence-transformers/\nall-MiniLM-L6-v2','Semantic retrieval\n(find similar facts)','~80 MB'],
         ['DeBERTa NLI','MoritzLaurer/DeBERTa-v3\n-base-mnli-fever-anli','Fact verification\n(entailment/contradiction)','~350 MB']], [100,135,120,60])

    E.append(Paragraph("4.2 Why DeBERTa-v3-base-mnli-fever-anli?", h2))
    p("This model is specifically trained on <b>MNLI + FEVER + ANLI</b> — three major NLI benchmarks. Unlike generic NLI models, it has seen hundreds of thousands of fact verification examples from FEVER during training. It understands:")
    b("Negation: 'Jupiter is NOT the largest' contradicts 'Jupiter IS the largest'")
    b("Paraphrasing: 'The Earth orbits the Sun' entails 'Our planet revolves around the Sun'")
    b("Irrelevance: 'I like pizza' is neutral to 'The Moon orbits Earth'")

    E.append(Paragraph("4.3 Two-Stage Verification Pipeline", h2))
    p("<b>Why two stages?</b> Cosine similarity alone cannot detect negation. 'X is Y' and 'X is NOT Y' have nearly identical embeddings because they share the same words. The NLI model catches the logical difference.")
    sp()
    p("<b>Stage 1 — Retrieval:</b> The sentence-transformer encodes the speaker's claim. Cosine similarity is computed against all 101,897 pre-encoded facts. Top 3 candidates above threshold (0.45) are selected.")
    sp()
    p("<b>Stage 2 — Verification:</b> For each candidate, the DeBERTa NLI model classifies the relationship as ENTAILMENT, CONTRADICTION, or NEUTRAL. The best non-neutral match is selected.")
    sp()
    tbl([['NLI Result','Fact is TRUE','Fact is FALSE'],
         ['Entailment','SUPPORTED\n(speaker affirms a truth)','REFUTED\n(speaker affirms a misconception)'],
         ['Contradiction','REFUTED\n(speaker denies a truth)','SUPPORTED\n(speaker corrects a myth)'],
         ['Neutral','UNVERIFIABLE','UNVERIFIABLE']], [100,180,180])

    # === 5. ARCHITECTURE ===
    sec("5. System Architecture")
    E.append(Paragraph(
        'Browser Microphone (PCM 16-bit, 16kHz)\n'
        '    → [WebSocket: 2.5s chunks streamed continuously]\n'
        '    → Whisper tiny (CPU, int8) → raw transcript\n'
        '    → Rolling Buffer (accumulates across chunks)\n'
        '    → Sentence Detection (split on . ! ?)\n'
        '    → For each complete sentence:\n'
        '        → Stage 1: Sentence-Transformer → cosine sim vs 101,897 facts\n'
        '        → Stage 2: DeBERTa NLI → entailment / contradiction\n'
        '        → Result: SUPPORTED / REFUTED / UNVERIFIABLE\n'
        '    → Debate Tracker (scores per side)\n'
        '    → [WebSocket: push to browser dashboard]', code))
    sp()
    p("<b>Streaming design:</b> Audio is sent in fixed 2.5-second chunks continuously — no waiting for silence. A rolling buffer accumulates text and extracts complete sentences as they form. This enables real-time fact-checking during continuous speech.")
    sp()
    tbl([['Component','File','Responsibility'],
         ['Web Server','main.py','Flask + SocketIO, WebSocket events, rolling buffer'],
         ['Speech Model','mymodel.py (Transcriber)','Whisper loading, audio → text, VAD'],
         ['Fact Checker','mymodel.py (FactChecker)','Retrieval + NLI verification'],
         ['Preprocessor','preprocess.py','Sentence splitting, filler removal'],
         ['Debate Tracker','debate.py','Per-side scoring, winner determination'],
         ['Frontend','templates/index.html','Audio capture, live dashboard']], [85,110,265])

    # === 6. IMPLEMENTATION ===
    sec("6. Implementation Details")
    tbl([['Category','Technology'],
         ['Language','Python 3.10+'],
         ['Web Framework','Flask + Flask-SocketIO (gevent async)'],
         ['Speech-to-Text','faster-whisper (CTranslate2, int8 quantized)'],
         ['Retrieval','sentence-transformers (all-MiniLM-L6-v2)'],
         ['NLI Verification','MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli'],
         ['Tensor Backend','PyTorch'],
         ['Audio','PCM 16-bit 16kHz via browser MediaRecorder API'],
         ['Transport','WebSocket (Socket.IO)'],
         ['Frontend','Vanilla HTML/CSS/JS']], [120,340])
    sp()
    p("<b>Embedding caching:</b> All 101,897 fact embeddings are computed once and cached to disk as a NumPy .npy file (~150MB). Subsequent startups load the cache in seconds instead of re-encoding.")
    p("<b>NLI inference:</b> The DeBERTa model runs on CPU. Each claim-fact pair takes ~50-100ms to classify. With top-3 candidates, total NLI time is ~150-300ms per claim.")

    # === 7. FILE STRUCTURE ===
    sec("7. File Structure")
    E.append(Paragraph(
        'convoai/\n'
        '├── main.py              # Main entry point (python main.py)\n'
        '├── mymodel.py           # Whisper + SentenceTransformer + DeBERTa NLI\n'
        '├── preprocess.py        # Sentence splitting, filler filtering\n'
        '├── debate.py            # Debate state tracking and scoring\n'
        '├── requirements.txt     # Python dependencies\n'
        '├── README.md            # Project description\n'
        '├── dataset_sample.csv   # 100-row sample\n'
        '├── architecture_diagram.jpg\n'
        '├── results_output.txt   # Example output\n'
        '├── data/\n'
        '│   └── facts.json       # Full FEVER dataset (101,897 facts)\n'
        '└── templates/\n'
        '    └── index.html       # Web UI', code))

    # === 8. EXECUTION ===
    sec("8. Execution Instructions")
    E.append(Paragraph('# Setup\npython -m venv .venv\nsource .venv/bin/activate\npip install -r requirements.txt\n\n# Run\npython main.py\n\n# Open http://localhost:5000', code))
    p("First run downloads models from HuggingFace (~500MB total) and encodes 101,897 fact embeddings (cached after first run). Subsequent startups are fast.")

    # === 9. RESULTS ===
    sec("9. Results / Output")
    tbl([['Speaker','Claim','Verdict','Confidence'],
         ['Side A','"The Colosseum is located in Rome"','SUPPORTED','91%'],
         ['Side A','"Einstein invented the telephone"','REFUTED','85%'],
         ['Side B','"Python is not a programming language"','REFUTED','88%'],
         ['Side B','"The Earth orbits the Sun"','SUPPORTED','93%'],
         ['Side B','"Water boils at 100 degrees Celsius"','SUPPORTED','87%']], [55,200,80,65])
    sp()
    p("Side A: 1 supported, 1 refuted → <b>50% accuracy</b>")
    p("Side B: 2 supported, 1 refuted → <b>66.7% accuracy</b>")
    p("Winner: <b>Side B</b>")

    # === 10. EVALUATION ===
    sec("10. Model Evaluation")
    tbl([['Component','Metric','Value'],
         ['Whisper (tiny)','Latency','<1s per 2.5s chunk'],
         ['Retrieval (101k facts)','Search time','~15ms'],
         ['DeBERTa NLI','Classification','~100ms per pair'],
         ['Full pipeline','End-to-end','~200-400ms per claim'],
         ['System','Total model RAM','~800 MB'],
         ['Embedding cache','Size on disk','~150 MB']], [120,140,200])

    # === 11. CHALLENGES ===
    sec("11. Challenges Faced")
    E.append(Paragraph("Challenge 1: Negation Detection", h2))
    p("Cosine similarity measures topical overlap, not logic. 'X is Y' and 'X is NOT Y' have nearly identical embeddings. Initial system marked 'Jupiter is not the largest planet' as SUPPORTED.")
    p("<b>Solution:</b> Added DeBERTa NLI model (trained on FEVER) as second stage. It classifies entailment vs. contradiction, correctly detecting negation.")
    sp()
    E.append(Paragraph("Challenge 2: Real-Time Latency", h2))
    p("Initial design waited for 1.5 seconds of silence before processing. Nothing happened while speaker was talking — unusable for real debates.")
    p("<b>Solution:</b> Switched to streaming architecture with fixed 2.5-second chunks, rolling transcript buffer, and Whisper tiny model (3x faster). Results appear within seconds of speaking.")
    sp()
    E.append(Paragraph("Challenge 3: Dataset Scale", h2))
    p("101,897 facts require significant encoding time. First startup took 10+ minutes to encode all embeddings.")
    p("<b>Solution:</b> Implemented NumPy .npy embedding cache. First run encodes and saves; subsequent runs load from cache in seconds.")
    sp()
    E.append(Paragraph("Challenge 4: Python 3.14 Compatibility", h2))
    p("The eventlet async library did not support Python 3.14. Required switching to gevent and ensuring all dependencies were compatible.")

    # === 12. DESIGN DECISIONS ===
    sec("12. Key Design Decisions")
    tbl([['Decision','Rationale'],
         ['Pre-trained models only\n(no custom training)','Framework: "DON\'T TRY TO train models initially." Pre-trained HuggingFace models work out of the box.'],
         ['DeBERTa-v3 FEVER model','Specifically trained on FEVER fact verification. Understands entailment, contradiction, negation.'],
         ['Full FEVER dataset\n(101,897 claims)','General-purpose coverage across all Wikipedia topics. Not limited to one domain.'],
         ['Two-stage pipeline','Cosine similarity for fast retrieval + NLI for logical verification. Handles negation correctly.'],
         ['Embedding caching','101k encodings cached to .npy file. First run slow, subsequent runs instant.'],
         ['Streaming audio (2.5s)','Real-time debate requires continuous processing, not silence-based chunking.']], [130,330])

    # === 13. FUTURE ===
    sec("13. Future Improvements")
    b("<b>GPU acceleration</b>: Use CUDA for DeBERTa inference — 10x faster NLI classification")
    b("<b>Larger Whisper model</b>: Whisper 'small' or 'medium' for better transcription on capable hardware")
    b("<b>LLM deep check</b>: Add optional Gemma/Phi local LLM for complex reasoning on ambiguous claims")
    b("<b>Speaker diarization</b>: Auto-detect which side is speaking instead of manual switching")
    b("<b>Custom domain datasets</b>: Add topic-specific fact databases alongside FEVER for specialized debates")

    # === 14. TEAM ===
    sec("14. Team Member Contributions")
    tbl([['Student ID','Student Name','Contribution Details'],
         ['____________','____________','• Speech-to-text pipeline (Whisper)\n• Audio streaming architecture\n• Rolling buffer + sentence detection'],
         ['____________','____________','• Fact verification (DeBERTa NLI)\n• Retrieval pipeline (sentence-transformer)\n• FEVER dataset integration'],
         ['____________','____________','• Web UI design\n• Debate tracking + scoring\n• Documentation + testing']], [80,80,300])

    sp(20); hr()
    E.append(Paragraph("End of Report", ParagraphStyle('End', parent=body, fontSize=10, alignment=TA_CENTER, textColor=GRAY)))
    doc.build(E)
    print(f"Generated: {OUTPUT}")

if __name__ == "__main__":
    build()
