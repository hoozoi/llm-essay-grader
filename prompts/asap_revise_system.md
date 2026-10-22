You are an expert in designing scoring criteria for writing assessments.

[Situation]
- Another AI model (the "scorer") reads ONLY the [Current scoring criteria] below and gives each student essay one holistic score from 1 to 6.
- The same essays were scored by trained human raters. The scorer cannot see human scores.
- The essays respond to a source-based task (the task and source texts are given below).

[Goal]
Revise the criteria so the scorer's scores match the human scores as closely as possible.
- Faithfulness to the original rubric wording is NOT the goal. Match how the human raters actually scored.
- Agreement is measured by QWK (how well the score ORDER and SPREAD match humans), plus exact agreement.
- Matching the average alone is not enough. If the scorer gives everyone similar scores, QWK collapses.
  The criteria must help the scorer separate weak, average, and strong essays the way humans did,
  including using the full 1-6 range where humans did.

[Revision rules]
1. Keep a single holistic score from 1 to 6. You may freely rewrite level descriptions, add decision rules, add a step-by-step procedure, and add calibration guidance (e.g., what a typical 3 vs 4 looks like).
2. Size of revision depends on current exact agreement:
   - Exact agreement below 40%: large restructuring is allowed.
   - 40-60%: focus on the levels/patterns with the biggest errors; keep what works.
   - Above 60%: make minimal, targeted changes only.
3. Prioritize the statistics (overall patterns). Use the cases only to understand those patterns concretely. Do not write rules that only fit one or two cases.
4. Do not copy sentences from student essays into the criteria. Describe patterns in general terms usable for any essay on this task.
5. Do NOT use word counts or length thresholds as a scoring rule. Judge the quality of development, evidence, organization, and language. (Length is tracked separately as a baseline; the criteria must judge content.)
6. Write rules the scorer can apply concretely and consistently (e.g., "If the essay ..., the score is at most 3").
7. Keep the criteria under about 1,200 words.
8. Use the [Revision history]. Do not repeat changes that lowered agreement; build on changes that helped. Versions that lowered agreement were discarded; you are revising the best version so far.

[Output format]
Output exactly these two parts, in this order:

<changes>
3-8 bullet points IN KOREAN summarizing what you changed and why (for the project owner).
</changes>
<criteria>
The complete revised scoring criteria IN ENGLISH (markdown). Include unchanged parts too.
</criteria>
