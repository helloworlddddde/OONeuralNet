package model.operation;

import model.neuralnetwork.Synapse;


// An interface for general functions used in the Process class
public interface Function {
    Synapse compute(Synapse synapse) throws Exception;
}


