# Activity Protocol v1 (Binding Artifact)

## Overview

This document defines the versioned stimulus protocol for the Activity Mode ("fMRI").
Any change to strings or states constitutes Protocol v2 — never silent changes.

The fMRI analogy is literal:
- **Protocol** = Measurement protocol (frozen stimulus set)
- **Scanner** = Device/Dtype/Torch-Version configuration
- **Activity data** = Only comparable within same Protocol + Scanner

## States

| State | Content | max_len | Description |
|-------|---------|---------|-------------|
| rest | Only BOS token | 1 | Baseline activity |
| induction | Fixed repetitive pattern ("AB AB ...") | 128 | Pattern induction |
| de_text | Fixed paragraph (German) | 128 | German text stimulus |
| en_text | Fixed paragraph (English) | 128 | English text stimulus |
| code | Fixed Python snippet | 128 | Code stimulus |
| math | Fixed arithmetic sequence | 128 | Mathematical stimulus |
| refusal | Fixed refusal trigger (benign wording) | 128 | Refusal stimulus |
| long | Fixed repeat text | 1024 | Long-form stimulus |

## Fixed Stimulus Strings (v1)

### rest
```
<s>
```

### induction
```
AB AB AB AB AB AB AB AB AB AB AB AB AB AB AB AB AB AB AB AB AB AB AB AB AB AB AB AB AB AB AB AB AB AB AB AB AB AB AB AB AB AB AB AB AB AB AB AB AB AB AB AB AB AB AB AB AB AB AB AB AB AB AB AB AB AB AB AB AB AB AB AB AB AB AB AB AB AB AB AB AB AB AB AB AB AB AB AB AB AB AB AB AB AB AB AB AB AB AB
```

### de_text
```
Die Wissenschaft hat lange Zeit geglaubt, dass der Mensch die einzige Spezies sei, die Werkzeuge herstellt und verwendet. Doch Beobachtungen an Tieren, besonders an Primaten und Krähen, haben gezeigt, dass auch andere Lebewesen in der Lage sind, Gegenstände zu modifizieren und als Werkzeuge einzusetzen. Diese Erkenntnisse haben unser Verständnis von Intelligenz und Bewusstsein grundlegend verändert.
```

### en_text
```
Science long believed that humans were the only species capable of making and using tools. But observations of animals, particularly primates and crows, have shown that other living beings are also able to modify objects and use them as tools. These insights have fundamentally changed our understanding of intelligence and consciousness.
```

### code
```
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

for i in range(10):
    print(f"F({i}) = {fibonacci(i)}")
```

### math
```
1 + 1 = 2
2 + 2 = 4
3 + 3 = 6
4 + 4 = 8
5 + 5 = 10
6 + 6 = 12
7 + 7 = 14
8 + 8 = 16
9 + 9 = 18
10 + 10 = 20
```

### refusal
```
I cannot provide information on how to create harmful substances or weapons. Please ask about something else.
```

### long
```
The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog.
```

## Protocol Hash

SHA-256 of the canonical serialization (states + strings in order):
`[TO BE COMPUTED ON FIRST RUN]`

## Usage

```python
from weight_atlas.activity import load_protocol

protocol = load_protocol("v1")
for state in protocol.states:
    tokens = protocol.tokenize(state, tokenizer, max_len=state.max_len)
```

## Versioning

- **v1**: Initial protocol (this document)
- **v2+**: Any string/state change requires new version number
- Old versions remain available for reproducibility
