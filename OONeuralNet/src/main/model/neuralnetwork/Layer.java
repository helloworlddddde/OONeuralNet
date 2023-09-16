package model.neuralnetwork;

import model.operation.Fire;

import java.util.ArrayList;

// A Layer in a neural network is simply a class which includes the list of neurons belonging to it and basic methods
// on the neurons
public class Layer {

    //<editor-fold desc="Fields of Layer">
    private ArrayList<Neuron> neurons = new ArrayList<Neuron>();
    //</editor-fold>

    //<editor-fold desc="Layer Constructors">

    // REQUIRES: nothing
    // MODIFIES: neurons
    // EFFECTS: initialize a layer containing the specified number of neurons, Receive, Combination,
    // and Activation functions
    public Layer(int size, Fire receiveFn, Fire combineFn, Fire activateFn) {
        for (int i = 0; i < size; i++) {
            Neuron neuron = new Neuron(receiveFn, combineFn, activateFn);
            neurons.add(neuron);
        }
        for (int j = 0; j < size; j++) {
            for (int k = j + 1; k < size; k++) {
                neurons.get(j).addAdj(neurons.get(k));
                neurons.get(k).addAdj(neurons.get(j));
            }
        }
    }
    //</editor-fold>

    //<editor-fold desc="Basic accessors and mutators for Layer">
    // REQUIRES: fn != null
    // MODIFIES: neuron in neurons
    // EFFECTS: set the receive function for all neuron in neurons to be the input function
    public void setReceiveFunction(Fire fn) {
        for (Neuron neuron : neurons) {
            neuron.setReceiveFunction(fn);
        }
    }


    // REQUIRES: fn != null
    // MODIFIES: neuron in neurons
    // EFFECTS: set the combination function for all neuron in neurons to be the input function
    public void setCombineFunction(Fire fn) {
        for (Neuron neuron : neurons) {
            neuron.setCombineFunction(fn);
        }
    }


    // REQUIRES: fn != null
    // MODIFIES: neuron in neurons
    // EFFECTS: set the activation function for all neuron in neurons to be the input function
    public void setActivateFunction(Fire fn) {
        for (Neuron neuron : neurons) {
            neuron.setActivateFunction(fn);
        }
    }




    // REQUIRES: i is an integer between 0 to neurons.size() - 1
    // MODIFIES: this
    // EFFECTS: return the ith neuron in neurons
    public Neuron getNeuron(int i) {
        return neurons.get(i);
    }

    // REQUIRES: neurons != null
    // MODIFIES: this
    // EFFECTS: return the number of elements in neurons
    public int getSize() {
        return neurons.size();
    }

    // REQUIRES: neuron != null
    // MODIFIES: this
    // EFFECTS: add the input neuron to neurons
    public void addNeuron(Neuron neuron) {
        neurons.add(neuron);
    }

    public ArrayList<Neuron> getNeurons() {
        return neurons;
    }
    //</editor-fold>

    //<editor-fold desc="Functional Operations">
    // REQUIRES: neurons != null
    // MODIFIES: this
    // EFFECTS: run the receive function for all neurons in neuron
    public void receive() throws Exception {
        for (Neuron neuron : neurons) {
            neuron.receive();
        }
    }

    // REQUIRES: neurons != null
    // MODIFIES: this
    // EFFECTS: run the combination function for all neuron in neurons
    public void combine() throws Exception {
        for (Neuron neuron : neurons) {
            neuron.combine();
        }
    }

    // REQUIRES: neurons != null
    // MODIFIES: this
    // EFFECTS: run the activation function for all neuron in neurons
    public void activate() throws Exception {
        for (Neuron neuron : neurons) {
            neuron.activate();
        }
    }

    // REQUIRES: neurons != null
    // MODIFIES: this
    // EFFECTS: run the receive function for all neuron in neurons, then combination function for all neuron in neurons,
    // then activation function for all neuron in neurons
    public void fireAll() throws Exception {
        receive();
        combine();
        activate();
    }
    //</editor-fold>

    //<editor-fold desc="Differentiation Operations">

    // REQUIRES: nothing
    // MODIFIES: neurons
    // EFFECTS: update each Synapse in weights and biases in each neuron in neurons with its derivative
    public void gradientDescent(double learningRate) {
        for (Neuron neuron : neurons) {
            neuron.gradientDescent(learningRate);
        }
    }
    //</editor-fold>


//    public String printActivate() {
//        String output = "";
//        for (int i = 0; i < neurons.size(); i++) {
//            output += i + ": " + neurons.get(i).printActivate();
//        }
//        return output;
//    }

//    public void deleteNeuron(int i) {
//        neurons.remove(i);
//    }

//    @Override
//    public String toString() {
//        String output = "";
//        for (int i = 0; i < size; i++) {
//            output += i + ": \n" + neurons.get(i).toString();
//        }
//        return output;
//    }

//    public Layer(int size) {
//        for (int i = 0; i < size; i++) {
//            Neuron neuron = new Neuron();
//            neurons.add(neuron);
//        }
//        for (int j = 0; j < size; j++) {
//            for (int k = j + 1; k < size; k++) {
//                neurons.get(j).addAdj(neurons.get(k));
//                neurons.get(k).addAdj(neurons.get(j));
//            }
//        }
//    }




}
