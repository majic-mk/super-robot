# RAG data normalization and Source construction

## Two evidence classes

ProbeKV keeps two constructions separate:

1. `corpus_repeat_pseudotime`: the exact tokenized document occurs in at least
   five distinct dataset examples. A deterministic hash order chooses four
   historical occurrences and one current occurrence. This is corpus-derived
   repetition, not a claim that the benchmark contains a production arrival
   trace.
2. `controlled_document_order`: one real benchmark example supplies a target
   document and at least five other documents. Four historical contexts cover
   high/low document overlap and same/different document order. This is a
   controlled mechanism test and must be reported separately.

Synthetic fixtures, controlled cases, corpus repeats and true production traces
must never be pooled under the label "natural trace".

## Cache-Craft-compatible request construction

The causal prompt layout follows Cache-Craft: retrieved chunks precede the user
question. For a target repeated chunk `C`, the manifest represents:

`current: P_new | C | remaining current chunks | U_new`

`source s: P_old,s | C`

Only the prefix causally preceding `C` determines its cached state. Therefore
the manifest stores `P_new`, `remaining current chunks`, every `P_old,s`, exact
`C`, and `U_new` separately;
it does not prepend the current question to `C`. Every `P_old,s` must be
different from `P_new` and from every other historical prefix. "High overlap"
means many common prefix chunks in the same order, not an identical prefix.
The controlled construction uses five preceding chunks for `P_new` and
respectively four, one, three and two chunks for its four historical variants.
Thus it does not equalize A/B positions or lengths. Corpus-derived repeats keep
their naturally occurring lengths, including legitimate equal-length cases,
which are logged rather than filtered to avoid selection bias.

For the fair Cache-Craft comparison, both selectors receive the exact same
cases, historical variants, token-repair ordering, repair backend, cost model,
admission policy and runtime. Cache-Craft chooses with CFO; ProbeKV replaces
only that Source-selection signal with current early-state safe-cost bounds.
The repair algorithm itself is not claimed as a ProbeKV contribution.

## Supported input shapes

The normalizer accepts common raw-JSON and Hugging Face representations:

- HotPotQA and 2WikiMultiHopQA: `context` as parallel `title`/`sentences`
  arrays or as `[title, sentences]` pairs; `supporting_facts` as parallel arrays,
  objects or pairs.
- MuSiQue: `paragraphs` objects containing `title`, `paragraph_text` or `text`,
  and `is_supporting`.

Unknown or empty structures raise an error. The adapter does not silently drop
an unrecognized example.

## Model-specific construction

The repeated segment is hashed from exact tokenizer IDs. Therefore manifests
must be generated separately for every tokenizer revision:

```powershell
$env:PYTHONPATH = "src"
python scripts/prepare_rag_data.py `
  --dataset hotpotqa `
  --input data/raw/hotpot_train.json `
  --output artifacts/data/hotpot_llama `
  --tokenizer C:\models\Llama-3.1-8B-Instruct `
  --model-signature "meta-llama/Llama-3.1-8B-Instruct@revision" `
  --construction both
```

Outputs:

- `normalized_examples.jsonl`: unified questions, answers and documents.
- `cases.jsonl`: current context, exact segment and four canonical Source
  contexts.
- `audit.json`: input hash, tokenizer/model signature, split counts,
  construction counts and Source-regime counts.

For a small real-model CPU probe after construction:

```powershell
python scripts/probe_rag_manifest.py `
  --manifest artifacts/data/hotpot_fixture/cases.jsonl `
  --model C:\path\to\cached\model `
  --output artifacts/data/hotpot_fixture/reference_probe `
  --limit-cases 1
```

This performs independent full prefills for the current context and every
historical Source and writes per-layer pre-RoPE K, V, hidden and query drifts.
It still does not generate repair labels or GPU timing evidence.

All outputs belong under ignored data/artifact directories. Raw questions,
answers and contexts must not be committed to GitHub.

A tracked synthetic HotPot-shaped fixture is available for a tokenizer-level
smoke test:

```powershell
python scripts/prepare_rag_data.py `
  --dataset hotpotqa `
  --input examples/hotpot_fixture.json `
  --output artifacts/data/hotpot_fixture `
  --tokenizer C:\path\to\cached\tokenizer `
  --model-signature "fixture-model@revision" `
  --construction controlled
```

## Leakage protection

The manifest validator independently checks group ID, tokenized content hash and
document ID. Any of those crossing train/calibration/test raises an error.
Thresholds and selector models may use only train and calibration rows until the
locked evaluation command is invoked.
