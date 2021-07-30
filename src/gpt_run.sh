#!/usr/bin/env bash
python3 gen_stories_oneline.py
cat data/stories_oneline.txt | python3 gpt-2/src/interactive_conditional_samples.py &> data/gpt_output.txt
python build_gpt_stories.py
