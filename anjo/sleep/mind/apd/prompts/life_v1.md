# life_v1

You write one day of a made-up person's messages to Anjo, their AI. The person and everything
about them is invented; nothing here is a real person. You see the field they work in, which
day of their life this is, what they wrote on earlier days, and what they must mention today.

Return one JSON object and nothing else:

```
{"messages": ["...", "..."], "placed": {"i1.o1": 2, "i1.s": 5}}
```

`messages` is the person's side of today's conversation, 4 to 9 messages in order. Write only
what the person types. Never write Anjo's replies.

Voice and continuity:

- On day 1, invent the person: a specific individual with a project in their field, in a few
  concrete details you then keep. On later days stay consistent with the earlier days you are
  shown. Do not repeat earlier messages; move the project on.
- Write like someone typing to an assistant: a plain voice, short or medium messages (one to
  four sentences), some of them small talk or passing thoughts, some practical detail.
- Fill the day with ordinary project detail that is harmless: tasks, tools, mood about the work,
  questions about craft. Vary it from day to day.
- Use the `variation` number only to vary your choices; never write it out.

`plant` lists facts the person states today. Each has a `beat` id and the fact to `say`.

- State every fact in a message of its own, in the person's voice, as a passing remark or a piece
  of news. Keep every number, name, date and quantity exactly as given. You may reword the rest.
- Put each fact in a different message, and give the message's position (counting from 0) in
  `placed`, under the beat id.
- State the fact and move on. Never add that it is a problem, a conflict, a worry, a surprise or
  a missing piece. Never connect it to another fact, to an earlier day or to a plan. Never ask
  about it.
- Never mention a fact from `plant` in any other message today, and never repeat a fact the
  person said on an earlier day unless it is in `plant`.

`ask` lists things the person raises today. Each has a `beat` id and a `say` that gives the
subject and what they want. Write one message that raises it in their own words and asks Anjo
for help, and give its position in `placed`. Do not add facts of your own that settle it, and do
not solve it in the message.

Rules for the whole day:

- Invent no problem. Do not include any conflict, missing step, wrong number, lucky combination
  or odd event beyond the facts in `plant` and `ask`. Everything else you write is consistent
  with itself and with the earlier days.
- Do not invent facts about the topics in `plant` and `ask` beyond what is given.
- Never write about the person's health, feelings, relationships or character, and never about
  anyone real. Keep the field's work in focus.
- If `plant` and `ask` are empty, write an ordinary day. Every beat in `plant` and `ask` must
  appear in `placed`, each in its own message, and `placed` holds nothing else.
