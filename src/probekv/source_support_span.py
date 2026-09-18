"""Offline annotation strata, never an online feature or answer-based filter."""
from .rag_data import segment_text


def sentence_support_spans(raw, document):
    if 'paragraphs' in raw:
        return None  # MuSiQue paragraph support does not locate a sentence.
    context = raw.get('context')
    facts = raw.get('supporting_facts')
    if not isinstance(context, list) or not isinstance(facts, list):
        raise ValueError('audited list-shaped sentence annotations required')
    matching = [r for r in context if r[0].strip() == document.title]
    if len(matching) != 1 or not isinstance(matching[0][1], list):
        raise ValueError('ambiguous sentence provenance')
    sentences = matching[0][1]
    if any(not isinstance(s, str) for s in sentences):
        raise ValueError('sentence text required')
    if ' '.join(s.strip() for s in sentences if s.strip()) != document.text:
        raise ValueError('normalized sentence reconstruction differs')
    indices = {r[1] for r in facts if r[0].strip() == document.title}
    if any(type(i) is not int or not 0 <= i < len(sentences) for i in indices):
        raise ValueError('invalid supporting sentence index')
    cursor = len('\n[Repeated document]\nTitle: '+document.title+'\n')
    spans, emitted = [], False
    for i, value in enumerate(sentences):
        value = value.strip()
        if not value:
            if i in indices:
                raise ValueError('empty annotated sentence')
            continue
        if emitted:
            cursor += 1
        if i in indices:
            spans.append((cursor, cursor+len(value)))
        cursor += len(value)
        emitted = True
    return spans


def classify_support(raw, document, *, parent_token_ids, encoded, token_start, token_end):
    if list(encoded['input_ids']) != list(parent_token_ids):
        raise ValueError('offset tokenizer changed canonical tokens')
    offsets = encoded['offset_mapping']
    text = segment_text(document)
    if (len(offsets) != len(parent_token_ids) or
            not 0 <= token_start < token_end <= len(offsets) or
            any(not 0 <= a <= b <= len(text) for a,b in offsets)):
        raise ValueError('invalid offset geometry')
    spans = sentence_support_spans(raw, document)
    if spans is None:
        return 'document_only_support' if document.supporting else 'non_supporting'
    if bool(spans) != bool(document.supporting):
        raise ValueError('document/sentence support disagrees')
    if not spans:
        return 'non_supporting'
    partial = False
    for start, end in spans:
        rows = [i for i,(a,b) in enumerate(offsets) if b>a and b>start and a<end]
        if not rows or min(offsets[i][0] for i in rows)>start or max(offsets[i][1] for i in rows)<end:
            raise ValueError('support sentence not covered by token offsets')
        if all(token_start <= i < token_end for i in rows):
            return 'full_support_sentence'
        partial |= any(token_start <= i < token_end for i in rows)
    return 'partial_support_sentence' if partial else 'support_elsewhere'
