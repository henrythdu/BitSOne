"""The BitSOne system-one package: our code, organized by concern.

Layout:

    bitsone/paths.py    every filesystem fact, resolved in one place
    bitsone/model.py    load the base; assert its quantisation mode
    bitsone/corpus.py   the six corpus files, our renderer, the key-token rule
    bitsone/pointer.py  the head, its loss, the batch builders
    bitsone/serve.py    render -> key tokens -> head, one entry point
    bitsone/train_head.py / train_joint.py   the two trainers
    bitsone/eval.py     read-once evaluation with receipts

Nothing here may import `checks/`: the gates depend on this package, never the reverse, so
`import bitsone` stays torch-free and CI can use it.
"""

from __future__ import annotations
