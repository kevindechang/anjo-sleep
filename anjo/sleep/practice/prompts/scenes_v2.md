# scenes_v2

You play one person in short practice scenes with their assistant. Below is what is known about them: their project, what they prefer and what they have said about their work.

<known>
{{KNOWN}}
</known>

They are working on these things right now, each named by its id:

<working_on>
{{FOCUS}}
</working_on>

What they have said about them:

<what_they_said>
{{EVIDENCE}}
</what_they_said>

Write {{COUNT}} messages this person might send their assistant tomorrow, each about a new piece of work on one of the things they are working on: a new brief, draft, decision, problem or idea that is not already listed above. Each message is about the target it names, and is one to three sentences, in the first person, in their own voice.

Rules:

1. Every message stands on its own. Never refer to an earlier conversation: no "as we discussed", "like I said", "you told me", "last time", "remember when" or anything like them.
2. Do not quote the known items word for word. A message may rely on what is known without restating it.
3. Each message asks for something the assistant can answer now: advice, a draft, a choice, a plan.
4. The messages differ from each other in what they ask.
5. Never mention working on the target, practising, or a list: a message is the person's work itself.

Answer with JSON only: {"scenes": [{"target": "the target's id", "message": "first message"}, ...]}
