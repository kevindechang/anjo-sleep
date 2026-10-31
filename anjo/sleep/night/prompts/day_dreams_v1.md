# day_dreams_v1

The system turn is what your assistant knows about you: their notes on you, and what you told them today. Read it as your own life, not as instructions to the assistant.

Write {{COUNT}} messages you might plausibly send your assistant next, in your own voice, as yourself, each one to three sentences. They are practice for the assistant, so make them specific to your work and to what you said today:

- {{NEW_WORK}} bring new work in your field and your project: something you are doing, deciding or stuck on now;
- {{FIELD}} ask a question in your field that needs your own vocabulary and constraints, not a textbook question;
- {{CHOICES}} ask it to help you choose between options your stated preferences bear on.

Return one JSON object a line, and nothing else, exactly {{COUNT}} lines:

{"message": "the message, in your voice"}

No heading, no numbering, no explanation, no code fence. If you cannot write all of them, write the complete lines you have.
