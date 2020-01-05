- Implemented a deep learning model (a BILSTM tagger) to detect code switching (language mixture) and return both a list of tokens and a list with one language label per token.
- To simplify our work was focussed on English and Spanish, so we were only needed to return for each token either 'en', 'es' or 'other'.

## Data ##
- For code switching we will focus on Spanish and English, and the data provided is derived from http://www.care4lang.seas.gwu.edu/cs2/call.html.
- This data is a collection of tweets, in particular we have three files for the training set and three for the validation set:

- offsets_mod.tsv
- tweets.tsv
- data.tsv

- The first file has the id information about the tweets, together with the tokens positions and the gold labels.
- The second has the ids and the actual tweet text.
- The third has the combination of the previous files, with the tokens of each sentence and the gold labels associated. More specifically, the columns are:
**offsets_mod.tsv:** {tweet_id, user_id, start, end, gold label}
**tweets.tsv:** {tweet_id, user_id, tweet text}
**data.tsv:** {tweet_id, user_id, start, end, token, gold label}

The gold labels can be one of three:
- en
- es
- other

For this task, we were required to implement a **BILSTM** tagger.

----------------------------------------------------------------------

## Approach ##
- 
- We tried to implement a BiLSTM model with character embeddings and see how our model performs for this task.
- To encode the character-level information, we will use character embeddings and a LSTM to encode every word to a vector.

## Data processing ##
- There were lines in the *_data.tsv files which had **"** as a token and was inhibittng the entire reading of file in 
pandas **read_csv** function.
- Hence we removed all the lines from both **train** as well as **dev** data files which had **"** in them.
- Keeping a tweet together as this will later adds to context if our model can learn that too.
- We created a list of list of tuples, in which each word/token was as a tuple with it's tag and inside a list which contains all the tuples of words from a single tweet.
 
-----------------------------------------------------------------------
## Results ##

- Our system achieved an accuracy of **96.2%** when trained and tested on the **train_data.tsv** file only.
- The confusion matrix for the result is as below:

    |     |  0  |  1  |  2  |
    |-----|--------|:------:|:------:|
    |  **0**  | 1816    |  45   |  26   |
    |  **1**  | 96     |   4651  | 70   |
    |  **2**  |   65  | 51    | 2545    |

- The Classification report is as below:

    |      | precision | recall | f1-score | support |
    | ---- |:---------:|:------:|:--------:|:-------:|
    |  **Other**  |     0.92  |    0.96|      0.94|      1887|
    |  **En**  |     0.98  |    0.97|      0.97|      4817|
    |  **Es**  |     0.96  |    0.96|      0.96|      2661|
   
   ------------------------------------------------------------------- 
    
## Final test result ##
- Our system achieved an accuracy of **96.5%** when trained on the **train_data.tsv** file and tested on **dev_data.tsv** file.
- The confusion matrix for the result is as below:

    |     |  0  |  1  |  2  |
    |-----|--------|:------:|:------:|
    |  **0**  | 17929    |  156   |  230   |
    |  **1**  | 876    |   45412  | 618   |
    |  **2**  |   796  | 469    | 24715    |

- The Classification report is as below:

    |      | precision | recall | f1-score | support |
    | ---- |:---------:|:------:|:--------:|:-------:|
    |  **Other**  |     0.91  |    0.98|      0.95|      18315|
    |  **En**  |     0.99  |    0.97|      0.98|      46906|
    |  **Es**  |     0.97  |    0.95|      0.96|      25980|
    
   ------------------------------------------------------------------- 

## Running the system ##
- Keep the train and test dataset similar to the format of **train_data.tsv** in the same directory as the **script_task3.py**.
- run the command **python3 script_task3.py train_data.tsv test_data.tsv** 
- It'll show two images, 1. The variation of loss and validation loss during training. 2. The confusion matrix image.
- At last will print the confusion matrix as well as classification report along with the accuracy of the madel.

---------------------------------------------------------------------