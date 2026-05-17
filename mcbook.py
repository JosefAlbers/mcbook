# Copyright 2026 J Joe

from __future__ import annotations
import textwrap
import re
import json
import difflib
from pathlib import Path
from typing import Any

class KB:
    def __init__(self, src_dir: str | Path | None = None, db_path: str | Path | None = None):
        self.system = '''You are a Research Assistant and Technical Editor. You process document histories and discussion threads formatted in XML.

### XML Format Specifications:
1. <comment id="X" parent="Y">: A single post in a discussion thread. 
   - "id" is the unique identifier for this comment.
   - "parent" is the ID of the comment it is replying to (or "None" for the root).
2. <document id="X">: Full content of document X. This is the authoritative current version.
3. <diff>: Edit history leading to the document above, chronological (oldest first).
   - Older revisions: single stat line per step, e.g. "v0 -> v1: +10 -2"
   - Most recent change: standard unified diff, no context lines:
       --- v1
       +++ v2
       @@ ... @@
       -removed line
       +added line
4. <document ref="ID" /> inside a comment means that comment is discussing that document version.
5. <branch id="X">: Represents a branching hierarchy of documents or comments. 
   - Useful for seeing all derivative works or replies stemming from a single point.

### Operational Logic:
- Reconstruct the conversation flow by following the "parent" tags.
- Cross-reference comment IDs with <document ref="ID" /> to understand what version is being discussed.
- Use the <diff> block to understand how the document evolved and how prior critiques were addressed.'''
        self.db: dict[str, dict[str, Any]] = {}
        self.source_dir = (Path(src_dir) if src_dir is not None else None)
        self.db_path = (Path(db_path) if db_path is not None else None)
        if self.db_path and self.db_path.exists():
            self.db = json.loads(self.db_path.read_text())
        if self.source_dir:
            for path in self.source_dir.rglob("*"):
                if any(part.startswith(".") for part in path.parts):
                    continue
                if not path.is_file():
                    continue
                if self.db_path and path == self.db_path:
                    continue
                rel = str(path.relative_to(self.source_dir))
                if rel in self.db:
                    continue
                self.db[rel] = {
                    "id": rel,
                    "parent": None,
                    "children": [],
                    "content": path.read_text(errors="ignore"),
                }
        try:
            from rcrlm import load, infer
        except ImportError:
            raise SystemExit("pip install rcrlm")
        m = load()
        def llm(prompt: str) -> str:
            raw = infer(prompt, **m, max_new_tokens=1024, stream=False)['out_str'][0] # □ no batching for now
            raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL)
            raw, _, _ = raw.partition('<|im_end|>')
            return raw.strip()
        self.llm = llm
        self.save()

    def __call__(self, content: str, parent: str | None = None, id: str | None = None) -> str:
        id = id or self._next_id()
        if parent is not None:
            if parent not in self.db:
                raise KeyError(f"parent does not exist: {parent}")
            self.db[parent]["children"].append(id)
        self.db[id] = {
            "id": id,
            "parent": parent,
            "children": [],
            "content": content,
        }
        self.save()
        return id

    def __repr__(self):
        return json.dumps(self.db, ensure_ascii=False, indent=2)

    def __getitem__(self, id: str) -> str:
        return self.db[id]['content']

    def get_branch(self, id: str, indent: bool = False) -> str:
        def branch_format(node: dict[str, Any]) -> str:
            content = node["content"].strip()
            children = "".join(branch_format(child) for child in node["children"])
            inner_body = content + ("\n" + children if children else "")
            indented_body = textwrap.indent(inner_body, "  " if indent else "")
            return (
                f'<branch id="{node["id"]}">\n'
                f'{indented_body}\n'
                f'</branch>\n'
            )
        tree_data = self.down(id)
        return branch_format(tree_data).strip()

    def get_discussion(self, id: str) -> str:
        chain = []
        curr = self.up(id)
        while curr:
            chain.append(curr)
            curr = curr["parent"]
        chain.reverse()
        xml_elements = []
        for node in chain:
            parent_id = node["parent"]["id"] if node["parent"] else "None"
            xml_elements.append(
                f'<comment id="{node["id"]}" parent="{parent_id}">\n'
                f'{node["content"].strip()}\n'
                f'</comment>'
            )
        return "\n".join(xml_elements)

    def get_revision(self, id: str) -> str:
        chain = []
        curr = self.up(id)
        while curr:
            chain.append(curr)
            curr = curr["parent"]
        chain.reverse()
        doc = f'<document id="{chain[-1]["id"]}">\n{chain[-1]["content"].strip()}\n</document>'
        if len(chain) == 1:
            return doc
        diff_lines = []
        for i in range(len(chain) - 1):
            a, b = chain[i], chain[i + 1]
            a_lines = a["content"].splitlines(keepends=True)
            b_lines = b["content"].splitlines(keepends=True)
            is_last_step = (i == len(chain) - 2)
            if is_last_step:
                diff_lines += list(difflib.unified_diff(
                    a_lines, b_lines,
                    fromfile=a["id"], tofile=b["id"],
                    lineterm="", n=0,
                ))
            else:
                raw = list(difflib.unified_diff(a_lines, b_lines, lineterm=""))
                added   = sum(1 for l in raw if l.startswith('+') and not l.startswith('+++'))
                removed = sum(1 for l in raw if l.startswith('-') and not l.startswith('---'))
                diff_lines.append(f'{a["id"]} -> {b["id"]}: +{added} -{removed}')
        diff_block = '<diff>\n' + '\n'.join(diff_lines) + '\n</diff>'
        return f'{doc}\n{diff_block}'

    def save(self) -> None:
        if self.db_path:
            self.db_path.write_text(repr(self))

    def up(self, id: str) -> dict[str, Any] | None:
        if id is None:
            return None

        node = self.db[id]
        return {
            "id": node["id"],
            "content": node["content"],
            "parent": self.up(node["parent"]),
        }

    def down(self, id: str) -> dict[str, Any]:
        node = self.db[id]
        return {
            "id": node["id"],
            "content": node["content"],
            "children": [
                self.down(child_id)
                for child_id in node["children"]
            ],
        }

    def _next_id(self) -> str:
        i = 1
        while True:
            id = f"c{i}"
            if id not in self.db:
                return id
            i += 1

    __contains__ = lambda self, x: x in self.db
    __len__ = lambda self: len(self.db)
    __iter__ = lambda self: iter(self.db)

SUBMIT = """Write a compelling tech blog post based on the technical concepts presented in the provided data.

CRITICAL INSTRUCTIONS:
- Extract ONLY the domain knowledge (the actual technology, methods, or science).
- DO NOT mention "XML", "reverse_diff", "documents", "versions", "paths", or anything about the data format.
- DO NOT refer to "the latest document" or "the author." 
- Write it as a standalone, natural article introducing these concepts to the world.
"""

REVIEW = """Act as a senior peer-reviewer for a top-tier AI conference (like ICML or ICLR).
Review the latest version of the document provided in the data.

Requirements:
1. Provide a "Strengths" and "Weaknesses" section.
2. Be technically specific. If the document mentions a specific formula or method (like ASVD or DobiSVD), critique the logic or the clarity of that specific part.
3. Check if the current version effectively addresses any concerns raised in the previous <comment> threads.
4. Output your review as a clean, professional critique. DO NOT mention XML tags or paths.
"""

REBUTT = """Act as the Author of the paper. Write a formal rebuttal to the latest reviewer comment.

Requirements:
1. Address the specific weaknesses pointed out by the reviewer.
2. Use the 'reverse_diff' in the data to point out EXACTLY what was improved in the latest version to satisfy the reviewer (e.g., "In the latest revision, we added differentiable truncation to address the concern about...").
3. Maintain a polite, professional, and evidence-based tone.
4. DO NOT mention XML technical terms (like 'CDATA' or 'path segments').
"""

REVISE = """Act as the Author of the paper. Revise the document to address the reviewer's concerns.

Requirements:
1. Address the specific weaknesses pointed out in the review.
2. Preserve everything the reviewer praised in the Strengths section.
3. Output ONLY the full revised document, no commentary or meta-text.
4. DO NOT include <document> or <xml> tags in your response.
"""

def _build_prompt(kb: KB, dis_id: str, rev_id: str, task: str) -> str:
    xml = '\n'.join([kb.get_discussion(dis_id), kb.get_revision(rev_id)]).strip()
    return f'{kb.system}\n\n### Task:\n{task}\n\n### Input Data:\n<xml>\n{xml}\n</xml>'

def setup(src_dir: str = 'src', src_doc: str = 'svd.md') -> tuple[KB, str, str]:
    src_doc_path = Path(src_dir)/src_doc
    if src_doc_path.exists():
        kb = KB(src_dir)
    else:
        kb = KB()
        kb("# SVD based LLM compression methods\n\n- ASVD: `WX = (WS)(S⁻¹X)` where `Sᵢᵢ ∝ ∑ⁿⱼ₌₁│Xᵢⱼ│`\n- ASVD+ ≈ SVDLLM ≈ BasisSharing ≈ DynamicRank: uses cholesky `SSᵀ=XXᵀ`\n    - BasisSharing uses X from full precision before any pruning\n    - DynamicRank is BasisSharing + entropy over singular values to determine rank for each weight\n- SVDLLMv2: uses 2-round SVD (once on XXᵀ and then on WS); and instead of cholesky, uses √(SVD(XXᵀ)) to whiten\n- DobiSVD: differentiable truncation; uses WX for A and for X⁻¹ did IPCA", id=src_doc)

    return kb, src_doc, None

def submit(kb: KB, rev_id: str, dis_id: str|None, task: str = SUBMIT) -> tuple[KB, str, str]:
    prompt = _build_prompt(kb, dis_id, rev_id, task)
    out = kb.llm(prompt)
    rev_id = kb(out, parent=None)
    dis_id = kb(f'<document ref="{rev_id}" />', parent=None)
    return kb, rev_id, dis_id

def revise(kb: KB, rev_id: str, dis_id: str, task: str = REVISE) -> tuple[KB, str, str]:
    prompt = _build_prompt(kb, dis_id, rev_id, task)
    out = kb.llm(prompt)
    rev_id = kb(out, parent=rev_id)
    dis_id = kb(f'<document ref="{rev_id}" />', parent=dis_id)
    return kb, rev_id, dis_id

def review(kb: KB, rev_id: str, dis_id: str, task: str = REVIEW) -> tuple[KB, str, str]:
    prompt = _build_prompt(kb, dis_id, rev_id, task)
    out = kb.llm(prompt)
    dis_id = kb(out, parent=dis_id)
    return kb, rev_id, dis_id

def rebutt(kb: KB, rev_id: str, dis_id: str, task: str = REBUTT) -> tuple[KB, str, str]:
    prompt = _build_prompt(kb, dis_id, rev_id, task)
    out = kb.llm(prompt)
    dis_id = kb(out, parent=dis_id)
    return kb, rev_id, dis_id

def test(): 
    kb, rev_id, dis_id = setup()
    kb, rev_id, dis_id = submit(kb, rev_id, dis_id)
    root_rev_id = rev_id
    root_dis_id = dis_id
    kb, rev_id, dis_id = review(kb, rev_id, dis_id)
    kb, rev_id, dis_id = revise(kb, rev_id, dis_id)
    kb, rev_id, dis_id = rebutt(kb, rev_id, dis_id)
    print()
    print('--- REV ---')
    rev = kb.get_revision(rev_id)
    print(rev)
    print()
    print('--- DIS ---')
    dis = kb.get_discussion(dis_id)
    print(dis)
    print()
    print('--- DOW.DIS ---')
    dow = kb.get_branch(root_dis_id, indent=True)
    print(dow)
    print()
    print('--- DOW.REV ---')
    dow = kb.get_branch(root_rev_id, indent=True)
    print(dow)

if __name__ == "__main__":
    test()
