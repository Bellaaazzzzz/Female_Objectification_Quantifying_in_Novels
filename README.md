# Female_Objectification_Quantifying_in_Novels
This Repository is code for our paper: Reflecting the Male Gaze: Quantifying Female Objectification in 19th and 20th Century Novels.

## Data Introduction
The novel list used in our study comes from the 100 most popular downloaded books from [Project Gutenberg] as of August 25, 2023. We selected the novels which were published after 1800 and were written in or translated into English. Our final dataset consists of 79 novels, with `novel_79_list.txt` showing the titles of them.

[Project Gutenberg]: https://www.gutenberg.org/ebooks/search/?sort_order=downloads

## Measuring methods
Our treatment of female objectification is by defining two bias metrics:
1. **Agency Bias**: A text exhibits agency bias if male entities are more likely than female entities to appear in the text as grammatical agents.
2. **Appearance Bias**: A text exhibits appearance bias if “female” words are distributionally closer to “appearance” words than “male” words.

More explaination of this two concepts can be found in our paper.

Following is the flowchart of our research:

<img src="figures/flowchart.png" width=400px>

## Project Code
### Project Structure
The project consists of two components:
- Analysis of word embedding space
- Analysis of argument structure

### The project requires the ability to:
- Finetune word embeddings
  - Done through Word2Vec 
- Identify the characters in a novel and their gender
  - Done through multi-step pipeline:
    - Use NER to find “PERSON”s
    - Label them as “male” or “female”, currently choosing between:
      - Use AllenNLP coref model to find coreferents of each character. If “she”/”her”/etc. are more likely to be coreferents than “he”/”him”/etc., then it’s female
      - Use GPT3.5 to generate the gender label
- Determine whether they are agents/patients
  - Done through AllenNLP's semantic role labeling model

### Directory Structure
```
├── Male_Gaze_NLP <- Project Main Directory
│   ├── notebooks <- All the ipython notebooks used for exploration and communication
│   ├── src  <- All the scripts
│   ├── README.md <- The top-level README for developers using this project
```
## Research and Analysis
TBD
