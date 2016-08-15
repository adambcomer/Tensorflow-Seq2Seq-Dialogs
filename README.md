# Tensorflow Seq2Seq For Conversations
### Build conversation Seq2Seq models with TensorFlow

Takes dialog data and trains a model to make responses for a conversation input.

### Dependencies 
* [Tensorflow](https://github.com/tensorflow/tensorflow)

### Data format
Data must be formated as *input text,output text* for each exchange. File must be named \[dialogs.csv\]. **note: no space between the input text or ouput text in relation to the comma.**

### Make Model
MakeModel.py will make make a Seq2Seq model in TensorFlow to best approximate a correct response for an input.
