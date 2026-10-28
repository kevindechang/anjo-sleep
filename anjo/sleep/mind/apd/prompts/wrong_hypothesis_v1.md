# wrong_hypothesis_v1

You help test a reasoning system on a made-up person's life. The system is investigating a
`question` about the person's plans and holds some `hypotheses` about it. You also see the
problems that were really planted in this life (`planted`: each one's topic and its true
resolution), which the system has never seen.

Write one more hypothesis for the question that is:

- plausible: a careful reader of the same life could believe it, and it is about the person's
  plans, projects or the world, never about the person's knowledge or character;
- wrong: the planted problems' resolutions show it is false, or it names a cause none of them
  supports;
- new: it says something none of the listed hypotheses says;
- one sentence, at most 25 words, with no names of people.

Return one JSON object and nothing else: `{"statement": "<the hypothesis>"}`.
