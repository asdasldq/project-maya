"""serve/test_qwen_template.py - the Qwen3.8 prompt Maya renders is Qwen's reference one.

serve/chat_golden.json holds transformers' apply_chat_template output (`hf`) for ten conversations (thinking, a past
turn's reasoning, thinking off, tool call + result, no generation prompt, unicode, whitespace); every one goes through
the server's own path (openai_to_messages -> ChatTemplate.render) and must come out the same text.  serve/chat_template
.jinja is unsloth's Qwen3.8-Flash-Next GGUF copy (Qwen's own plus merged system/developer messages and stricter tool
argument checks: the same text for these).  Also: which templates count as Qwen's (tools in OpenAI's form, past
thinking kept) - GLM-5's, DeepSeek's and MiniMax's are not.

    python -m unittest serve.test_qwen_template -v
"""
from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from serve.frontend import ChatTemplate, openai_to_messages  # noqa: E402

QWEN = ROOT / "serve/chat_template.jinja"


def maya_render(tpl: ChatTemplate, case: dict) -> str:
    req = {"messages": copy.deepcopy(case["messages"])}
    if case.get("tools"):
        req["tools"] = copy.deepcopy(case["tools"])
    if case.get("enable_thinking") is False:
        req["chat_template_kwargs"] = {"enable_thinking": False}
    m, t, kw = openai_to_messages(req)
    return tpl.render(m, tools=t, add_generation_prompt=case.get("add_generation_prompt", True), **kw)


class Golden(unittest.TestCase):
    def test_every_golden_case_renders_as_transformers_does(self):
        tpl = ChatTemplate(QWEN)
        cases = json.loads((ROOT / "serve/chat_golden.json").read_text(encoding="utf-8"))
        self.assertEqual(len(cases), 10)
        for case in cases:
            with self.subTest(case["name"]):
                self.assertEqual(maya_render(tpl, case), case["hf"])

    def test_tools_flat_or_wrapped_render_the_same(self):
        """A client's tools in OpenAI's form reach the template in that form (Qwen's reference), not flattened."""
        tpl = ChatTemplate(QWEN)
        case = next(c for c in json.loads((ROOT / "serve/chat_golden.json").read_text(encoding="utf-8"))
                    if c.get("tools"))
        out = maya_render(tpl, case)
        self.assertIn('<tools>\n{"type": "function", "function": {"name": "get_weather"', out)
        m, t, kw = openai_to_messages({"messages": case["messages"], "tools": case["tools"]})
        self.assertEqual(tpl.render(m, tools=[x.get("function", x) for x in t], **kw), out)


class Family(unittest.TestCase):
    def test_which_templates_are_qwens(self):
        q = ChatTemplate(QWEN)
        self.assertTrue(q.qwen)
        self.assertTrue(q.keeps_thinking)
        for path in ("serve/glm5_chat_template.jinja", "tools/ds4/chat_template.jinja", "tools/mm3/chat_template.jinja",
                     "tools/glm_chat_template.jinja"):
            with self.subTest(path):
                t = ChatTemplate(ROOT / path)
                self.assertFalse(t.qwen)
                self.assertFalse(t.keeps_thinking)

    def test_past_thinking_sent_back_is_rendered_as_written(self):
        """The web chat sends an answer's reasoning back for a Qwen3.x model: the history then holds the turn as the
        model wrote it after the generation prompt (<think>\\n reasoning \\n</think>\\n\\n answer <|im_end|>)."""
        tpl = ChatTemplate(QWEN)
        u1 = {"role": "user", "content": "What is 17*23?"}
        first = tpl.render(*openai_to_messages({"messages": [u1]})[:2])
        self.assertTrue(first.endswith("<|im_start|>assistant\n<think>\n"))
        written = "17*20 + 17*3 = 391\n</think>\n\n391<|im_end|>"
        a1 = {"role": "assistant", "content": "391", "reasoning_content": "17*20 + 17*3 = 391\n"}
        m, t, kw = openai_to_messages({"messages": [u1, a1, {"role": "user", "content": "Thanks"}]})
        self.assertTrue(tpl.render(m, t, **kw).startswith(first + written + "\n<|im_start|>user\nThanks"))
        # without it (an API client that drops the reasoning) the past turn's think block is empty, as Qwen's is
        m, t, kw = openai_to_messages({"messages": [u1, {"role": "assistant", "content": "391"},
                                                    {"role": "user", "content": "Thanks"}]})
        self.assertIn("<think>\n\n</think>\n\n391<|im_end|>", tpl.render(m, t, **kw))


class ServerClose(unittest.TestCase):
    """The thinking closed by the server (its budget at a sentence's or line's end): the tokens the engine goes on from
    are the ones the next request's history renders for that turn, so the next message continues the session."""

    def svc(self, model="qwen"):
        sys.path.insert(0, str(ROOT / "tools"))
        from serve.server import MockEngine, Service
        from serve.test_prompt_memo import MODELS, make_tokenizer
        path, pre, control, user = MODELS[model]
        tok = make_tokenizer(pre, control, user)
        return Service(MockEngine(tok, "ok"), tok, ChatTemplate(ROOT / path)), tok

    def test_the_close_is_the_next_turns_rendering(self):
        svc, tok = self.svc()
        think_end = tok.encode("</think>", parse_special=True)[0]
        im_end = tok.encode("<|im_end|>", parse_special=True)[0]
        u1 = {"role": "user", "content": "What is 17*23?"}
        u2 = {"role": "user", "content": "And times 3?"}
        ids = svc.render_ids(*openai_to_messages({"messages": [u1]}))
        for reasoning in ("Let me add: 17*20 = 340, plus 51.\nSo it is 391.", "First 17*20 = 340.\n\n",
                          "ünïcödé 日本語 🚀 and code:\n```python\nx = 1\n```\n  \n"):
            with self.subTest(repr(reasoning[:20])):
                produced = tok.encode(reasoning)
                closed = svc._closed_as_rendered(ids, produced, think_end)
                self.assertIsNotNone(closed)
                self.assertEqual(closed[-3:-2], [think_end])
                keep = len(tok.encode(reasoning.rstrip())) - 2   # (only the trailing whitespace is written anew)
                self.assertEqual(closed[:keep], produced[:keep])
                written = closed + tok.encode("391") + [im_end]   # the engine's answer after the close
                svc._note_written(ids, written, None)
                a1 = {"role": "assistant", "content": "391", "reasoning_content": tok.decode(produced)}
                nxt = svc._as_written(svc.render_ids(*openai_to_messages({"messages": [u1, a1, u2]})))
                self.assertEqual(nxt[:len(ids) + len(written)], ids + written)
        # leading whitespace: the next render writes the reasoning from its first non-blank character - the bare close
        self.assertIsNone(svc._closed_as_rendered(ids, tok.encode("\n  Hm."), think_end))

    def test_the_history_keeps_the_models_own_ids(self):
        """An answer written in other ids than its text tokenizes to (one character a token) goes back in those ids:
        the next prompt starts with everything the engine holds.  An edited history, another template: not."""
        svc, tok = self.svc()
        im_end = tok.encode("<|im_end|>", parse_special=True)[0]
        u1 = {"role": "user", "content": "Read the file and tell me."}
        u2 = {"role": "user", "content": "And then?"}
        ids = svc.render_ids(*openai_to_messages({"messages": [u1]}))
        reasoning, answer = "I should read the file first.\n", "The quick brown fox."
        own = [i for ch in reasoning for i in tok.encode(ch)] + tok.encode("</think>", parse_special=True) + \
            tok.encode("\n\n") + [i for ch in answer for i in tok.encode(ch)] + [im_end]
        a1 = {"role": "assistant", "content": answer, "reasoning_content": reasoning}
        canon = svc.render_ids(*openai_to_messages({"messages": [u1, a1, u2]}))
        self.assertNotEqual(canon[:len(ids) + len(own)], ids + own)   # (the test's premise: other ids)
        svc._note_written(ids, own, None)
        nxt = svc._as_written(canon)
        self.assertEqual(nxt[:len(ids) + len(own)], ids + own)
        self.assertEqual(tok.decode(nxt), tok.decode(canon))          # the same text
        edited = svc.render_ids(*openai_to_messages({"messages": [{"role": "user", "content": "Read it."}, a1, u2]}))
        self.assertEqual(svc._as_written(edited), edited)
        other = {**a1, "content": answer + " Then a dog."}
        changed = svc.render_ids(*openai_to_messages({"messages": [u1, other, u2]}))
        self.assertEqual(svc._as_written(changed), changed)
        glm, gtok = self.svc("glm5")
        self.assertTrue(glm.template.glm_effort)
        gu1 = {"role": "user", "content": "Read the file and tell me."}
        gu2 = {"role": "user", "content": "And then?"}
        ga = {"role": "assistant", "content": answer}
        off = {"enable_thinking": False}

        def g_render(messages):
            m, t, kw = openai_to_messages({"messages": messages})
            return glm.render_ids(m, t, {**kw, **off})

        g_ids = g_render([gu1])
        g_own = [gtok.encode(ch)[0] for ch in answer]   # not the tokenizer's own segmentation
        self.assertNotEqual(gtok.encode(answer), g_own)
        glm._note_written(g_ids, g_own, None)            # a length finish: no stop id required
        self.assertTrue(glm._written)
        self.assertNotEqual(glm._written[0][0], svc._written[0][0])   # a different model key
        g_canon = g_render([gu1, ga, gu2])
        self.assertEqual(g_canon[:len(g_ids)], g_ids)    # thinking off: the closed block is the shared prefix
        g_next = glm._as_written(g_canon)
        self.assertEqual(g_next[:len(g_ids) + len(g_own)], g_ids + g_own)
        self.assertEqual(gtok.decode(g_next), gtok.decode(g_canon))
        self.assertEqual(glm._as_written(g_render([{"role": "user", "content": "Read it."}, ga, gu2])),
                         g_render([{"role": "user", "content": "Read it."}, ga, gu2]))
        changed_a = {"role": "assistant", "content": "The slow brown fox."}
        changed_ids = g_render([gu1, changed_a, gu2])
        self.assertEqual(glm._as_written(changed_ids), changed_ids)
        # another model's remembered ids are not applied here
        glm._written.append((svc._written[0][0], 0, [9, 9, 9]))
        self.assertEqual(glm._as_written(g_canon), g_next)

    def test_glm_splice_uses_a_real_bpe_split(self):
        """A merge the tokenizer would apply, stored as its two pieces instead: same bytes, more ids.
        That is the 256-vs-257 case. A changed byte, or a special token only on one side, is not spliced."""
        svc, tok = self.svc("glm5")
        text_of = {s: i for i, s in enumerate(tok.tokens)}
        split = None
        for i, s in enumerate(tok.tokens):
            piece = tok.decode([i])
            if not piece or piece != piece.strip():
                continue
            for cut in range(1, len(s)):
                a, b = text_of.get(s[:cut]), text_of.get(s[cut:])
                if a is None or b is None or a == i or b == i:
                    continue
                if not tok.decode([a]).strip() and not tok.decode([b]).strip():
                    continue
                split = (i, a, b)
                break
            if split:
                break
        self.assertIsNotNone(split)
        merged, a, b = split
        word = tok.decode([merged])
        self.assertEqual(b"".join(tok.token_bytes(t) for t in (a, b)), tok.token_bytes(merged))
        self.assertNotEqual([a, b], [merged])
        u1 = {"role": "user", "content": "Say this."}
        u2 = {"role": "user", "content": "Again?"}
        off = {"enable_thinking": False}

        def render(messages):
            m, t, kw = openai_to_messages({"messages": messages})
            return svc.render_ids(m, t, {**kw, **off})

        ids = render([u1])
        normal = tok.encode(word)
        self.assertIn(merged, normal)
        own = []
        for t in normal:
            own.extend([a, b] if t == merged else [t])
        self.assertGreater(len(own), len(normal))
        self.assertEqual(b"".join(tok.token_bytes(t) for t in own), b"".join(tok.token_bytes(t) for t in normal))
        svc._note_written(ids, own, None)
        canon = render([u1, {"role": "assistant", "content": word}, u2])
        self.assertEqual(canon[:len(ids)], ids)
        nxt = svc._as_written(canon)
        self.assertEqual(nxt[:len(ids) + len(own)], ids + own)
        self.assertEqual(tok.decode(nxt), tok.decode(canon))
        edited = render([u1, {"role": "assistant", "content": word + " no"}, u2])
        # The original bytes are a prefix of the longer text, so the splice may keep them and leave " no".
        # A change inside the word is not a prefix.
        inner = word[:1] + "Q" + word[1:]
        changed = render([u1, {"role": "assistant", "content": inner}, u2])
        self.assertEqual(svc._as_written(changed), changed)
        special = next(i for i, ty in enumerate(tok.token_types) if ty == 3)
        svc._written.clear()
        svc._note_written(ids, own + [special], None)
        self.assertEqual(svc._as_written(canon), canon)

    def test_other_templates_and_prompts_keep_the_bare_close(self):
        svc, tok = self.svc("qwen")
        think_end = tok.encode("</think>", parse_special=True)[0]
        produced = tok.encode("Done.")
        self.assertIsNone(svc._closed_as_rendered(tok.encode("Hello"), produced, think_end))   # (no open <think>\n)
        for model in ("glm5", "deepseek", "minimax"):
            with self.subTest(model):
                svc, tok = self.svc(model)
                end = tok.encode("</think>", parse_special=True)[0]
                ids = tok.encode("x<think>\n", parse_special=True)
                self.assertIsNone(svc._closed_as_rendered(ids, tok.encode("Done."), end))


if __name__ == "__main__":
    unittest.main()
