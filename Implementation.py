import tflearn

def create_dnn(num_layers, info):

    #create a deep neural network with num_layers layers, info all layers is
    #contained within info parameter, info : is a json dict

    input_shape = info['input_data']['shape']

    net = tflearn.input_data(shape = input_shape, dtype = info['input_data']['dtype'])

    layers = info['layers']
    layer_index = 0
    for layer in layers:
        layer_info = layers[layer_index]['layer'+str(layer_index+1)]
        net = tflearn.fully_connected(incoming = net, n_units = layer_info['units'], activation = layer_info['activation'])
        layer_index = layer_index + 1
    
    net = tflearn.regression(incoming = net, optimizer = "adam")

    net = tflearn.DNN(net)

    return net
    
    
