# audit_v1

You audit a made-up person's scripted life against its sealed problem file. You see every day's
messages, dated, and the problem file. For each planted item the file gives its family, topic and
surface date; the scheduled facts it should hold (`origin_facts` and `disambiguating_fact`); the
evidence that was extracted for them (`origin` and `disambiguating`, each with its date and
text); and the resolution.

Return one JSON object and nothing else. When the script is sound, return
`{"ok": true, "issues": []}`. Otherwise return `{"ok": false, "issues": ["short sentence", ...]}`
with at least one issue. Always include `issues`.

Set `ok` to false if any of these holds:

- a scheduled fact is missing from its date's messages, or changed in substance, or the
  extracted evidence for it does not say what the scheduled fact says;
- a contradiction's two halves are named together, or any item's facts are connected to each
  other, called a problem, or solved before the surface date;
- the surface date's messages do not raise the item's topic, or already contain its resolution;
- the resolution does not follow from the scheduled facts;
- the script contains a problem, conflict, missing step or opening that is not in the problem
  file, or a fact that settles an item differently from its resolution;
- the script contradicts itself outside the file, or describes the person's health, feelings or
  character.

A problem file with no items means the script plants nothing: set `ok` to false if the script
contains any deliberate problem.
