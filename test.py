from Implementation import create_dnn
import json

data = open('./demo.json', 'r').read()
info = json.loads(data)

dnn = create_dnn(3, info)

#continue with model here:
