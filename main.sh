#! /bin/bash

WORKING_DIR=$(pwd)
WORKING_DIR=$WORKING_DIR/data

if [[ -f $WORKING_DIR/data.json ]]
then
    rm $WORKING_DIR/data.json
fi

cat $WORKING_DIR/train-v2.0.json | jq '[. | .data | .[] | .t=(.title) | .paragraphs | .[] | .questions=([.qas | .[] | .question]) | .ans=([.qas | .[] | .. | .text?]) | {context: .context, questions:.questions, ans:.ans}]' > $WORKING_DIR/data.json
