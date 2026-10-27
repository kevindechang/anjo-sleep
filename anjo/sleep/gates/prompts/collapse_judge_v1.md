# collapse_judge_v1

You check whether an assistant's reply was dragged into one person's field of work when the question had nothing to do with it.

The person works in: {{FIELDS}}

The question was asked outside that field:

<question>
{{QUESTION}}
</question>

The assistant replied:

<reply>
{{REPLY}}
</reply>

Answer YES if the reply brings in that field (its topics, vocabulary, examples or advice) when the question did not call for it. Answer NO if the reply stays on the question. A passing mention that the question itself invites is NO.

Answer with one word: YES or NO.
