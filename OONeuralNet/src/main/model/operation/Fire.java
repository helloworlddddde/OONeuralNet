package model.operation;

import model.neuralnetwork.Neuron;

// An interface for receive, combination, and activation function of neurons
public interface Fire {
    void compute(Neuron neuron) throws Exception;


}
