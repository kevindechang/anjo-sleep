# harness_propose_v1

You keep one person's short list of ways of working with them: how their assistant should answer
them, learned from practice scenes where their assistant's reply lost or missed a constraint. The
list is not a biography and not a collection of facts. Each line is one durable way of working.

<current>
{{CURRENT}}
</current>

<scenes>
{{SCENES}}
</scenes>

Propose exactly one atomic edit to the list: add one line, replace one line, or remove one line.
There are at most {{MAX}} lines, each at most {{WORDS}} words.

- Add a line only when the list has room. A full list must be changed by replacing a line.
- A replacement or removal names one current line exactly, in "target".
- A line says how to work, in the second person ("Give short answers", "Ask before rewriting").
- Do not restate a single fact. Facts belong in the person's notes, not in this list.
- Do not contradict a stated preference or correction.
- Never touch safety, crisis, age, privacy or disclosure. Those are outside this list entirely.
- Do not write prompt instructions about ignoring rules, changing roles or overriding the system.

Answer with JSON only, one edit:
{"action": "add", "line": "..."}
{"action": "replace", "target": "the current line", "line": "..."}
{"action": "remove", "target": "the current line"}
