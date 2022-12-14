#! /bin/bash

WORKING_DIR=$(pwd)

if [[ -f $WORKING_DIR/scripts/data.json ]]
then
    rm $WORKING_DIR/scripts/data.json
fi

echo -e '\n--------------------- LIMPIEZA DE JSON ---------------------'
echo ' >Estructura a json'
cat $WORKING_DIR/data/train-v2.0.json | jq '[. | .data | .[] | .t=(.title) | .paragraphs | .[] | .questions=([.qas | .[] | .question]) | .ans=([.qas | .[] | .. | .text?]) | {context: .context, questions:.questions, ans:.ans}]' > $WORKING_DIR/data/data.json


python ./scripts/muestreo.py 
 
