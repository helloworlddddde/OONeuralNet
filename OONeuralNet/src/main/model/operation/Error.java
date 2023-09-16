package model.operation;

import model.tensor.Matrix;
import model.neuralnetwork.Layer;

// An interface for calculating error / loss function for the network
public interface Error {
    Matrix compute(Layer layer, Matrix[] matrix);
}
