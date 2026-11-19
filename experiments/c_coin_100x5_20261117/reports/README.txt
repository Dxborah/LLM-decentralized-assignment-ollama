All results cover the same 100 worlds, case_1 through case_100, with 5 actual runs per world.

MeanSuccessfulSteps excludes failed tasks. MeanStepsIncluding100StepFailures retains the recorded 100-step failure caps.

OptimalTrials counts successful trials whose Steps equal TrueOptimal (assignment makespan). The legacy Optimal column in raw data is a different metric.

Mean TrueOptimal across all 100 worlds: 13.17

The LLM method is matching game, prompt C, two agreement rounds, then a fair coin among contestants; base seed 20261117.

Each paired rule control reuses the exact saved initial rankings for its corresponding model/case/trial. Lowest competing agent ID wins each conflict. No new LLM calls.

The separate greedy method uses no LLM and reranks using shortest-path distances. Its five deterministic repeats are not independent random samples.

Coin percentage uses conflict count as denominator, not trial count. Failure counts describe task outcomes, not pipeline errors.

Gemma4 12B uses thinking disabled and 2048 output tokens. Llama3.1 uses 2048. Llama3.2 preserves 185 early uncapped trials and then uses 2048.

Ministral 3 8B and Phi-4 use a 4096 output cap and opt-in acceptance of literal control characters in otherwise complete ranking JSON; malformed/truncated responses remain invalid.

Qwen2.5-VL and Gemma4 E4B retain their original decoding profiles. These are comparisons of model configurations; within each paired rule comparison, initial rankings are held identical.

Earlier mean 12.69 for Gemma4 12B covered only the first 23 worlds (115 trials). This report covers all 100 (500 trials).

DeepSeek and other rejected or superseded pilots are excluded from these production results.
