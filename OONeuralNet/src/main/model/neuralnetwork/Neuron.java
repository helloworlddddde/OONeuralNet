package model.neuralnetwork;

import model.tensor.Matrix;
import model.operation.Fire;

import java.util.ArrayList;

// In a neural network, the neuron represents the receiver, processor, and sender of data (synapses). Each neuron holds
// weights and biases as matrices which are to be updated using automatic differentiation (for gradient descent).
// A neuron processes the input data in three steps, Receive, Combine, and Activate. Each neuron holds a(n) (Array)list
// reference to adjacent neurons (in same layer), previous neurons (in the previous layer),
// and next neurons (in then next layer).
public class Neuron {
    private ArrayList<Neuron> adj = new ArrayList<Neuron>();
    private ArrayList<Neuron> prev = new ArrayList<Neuron>();
    private ArrayList<Neuron> next = new ArrayList<Neuron>();
    private Matrix receiveData;
    private Matrix combineData;
    private Matrix activateData;
    private Fire receiveFunction;
    private Fire combineFunction;
    private Fire activateFunction;
    private Matrix weights;
    private Matrix biases;

    //<editor-fold desc="Neuron Constructors">
    // REQUIRES: nothing
    // MODIFIES: this
    // EFFECTS: create a new Neuron object with the specified Receive, Combination, and Activation function
    public Neuron(Fire receiveFunction, Fire combineFunction, Fire activateFunction) {
        this.receiveFunction = receiveFunction;
        this.combineFunction = combineFunction;
        this.activateFunction = activateFunction;
    }

    // REQUIRES: nothing
    // MODIFIES: this
    // EFFECTS: create a Neuron object without any synapses, weight, or functions
    public Neuron() {
    }
    //</editor-fold>

    //<editor-fold desc="Functional Operations">
    // REQUIRES: receiveFunction != null
    // MODIFIES: this
    // EFFECTS: run the Neuron's Receive function
    public void receive() throws Exception {
        receiveFunction.compute(this);
    }


    // REQUIRES: combineFunction != null
    // MODIFIES: this
    // EFFECTS: run the Neuron's Combination function
    public void combine() throws Exception {
        combineFunction.compute(this);
    }

    // REQUIRES: activateFunction != null
    // MODIFIES: this
    // EFFECTS: run the Neuron's Activation function
    public void activate() throws Exception {
        activateFunction.compute(this);
    }
    //</editor-fold>

    //<editor-fold desc="Basic accessors and mutators for Neuron">
    public void setCombineFunction(Fire function) {
        this.combineFunction = function;
    }

    public Matrix getActivateData() {
        return activateData;
    }

    public void setActivateData(Matrix activateData) {
        this.activateData = activateData;
    }

    public Matrix getCombineData() {
        return combineData;
    }

    public void setCombineData(Matrix combineData) {
        this.combineData = combineData;
    }

    public Matrix getReceiveData() {
        return receiveData;
    }

    public void setReceiveData(Matrix receiveData) {
        this.receiveData = receiveData;
    }

    public void setReceiveFunction(Fire function) {
        this.receiveFunction = function;
    }

    public void setActivateFunction(Fire function) {
        this.activateFunction = function;
    }

    public Matrix getWeights() {
        return weights;
    }

    public void setWeights(Matrix weights) {
        this.weights = weights;
    }

    public Matrix getBiases() {
        return biases;
    }

    public void setBiases(Matrix biases) {
        this.biases = biases;
    }

    public ArrayList<Neuron> getPrev() {
        return prev;
    }

    // REQUIRES: neuron != null
    // MODIFIES: prev
    // EFFECTS: add neuron to prev
    public void addPrev(Neuron prev) {
        this.prev.add(prev);
    }

    public ArrayList<Neuron> getAdj() {
        return adj;
    }

    // REQUIRES: i is an integer between 0 and adj.size()-1
    // MODIFIES: this
    // EFFECTS: return the ith neuron in adj
    public Neuron getAdj(int i) {
        return adj.get(i);
    }

    // REQUIRES: neuron != null
    // MODIFIES: adj
    // EFFECTS: add neuron to adj
    public void addAdj(Neuron neuron) {
        this.adj.add(neuron);
    }

    public ArrayList<Neuron> getNext() {
        return next;
    }

    // REQUIRES: neuron != null
    // MODIFIES: next
    // EFFECTS: add neuron to next
    public void addNext(Neuron next) {
        this.next.add(next);
    }
    //</editor-fold>

    //<editor-fold desc="Differentiation Operations">

    // REQUIRES: weights != null && biases != null
    // MODIFIES: weights and biases
    // EFFECTS: update each Synapse in weights and biases using its derivative
    public void gradientDescent(double learningRate) {
        int dimRow = weights.getDimRow();
        int dimCol = weights.getDimCol();
        for (int r = 0; r < dimRow; r++) {
            for (int c = 0; c < dimCol; c++) {
                weights.getSynapse(r, c).gradientDescent(learningRate);
            }
        }
        dimRow = biases.getDimRow();
        dimCol = biases.getDimCol();
        for (int r = 0; r < dimRow; r++) {
            for (int c = 0; c < dimCol; c++) {
                biases.getSynapse(r, c).gradientDescent(learningRate);
            }
        }
    }

//    public Fire getReceiveFunction() {
//        return receiveFunction;
//    }
//
//    public Fire getCombineFunction() {
//        return combineFunction;
//    }
//
//    public Fire getActivateFunction() {
//        return activateFunction;
//    }
    //</editor-fold>



//    public String printActivate() {
//        return activate.toString();
//    }


//    @Override
//    public String toString() {
//        String dweight = "[ ";
//        int dimRow = weights.getDimRow();
//        int dimCol = weights.getDimCol();
//        for (int r = 0; r < dimRow; r++) {
//            for (int c = 0; c < dimCol; c++) {
//                dweight += weights.getData(r, c).getDerivative() + " ";
//            }
//        }
//        dweight += " ]";
//        dimRow = biases.getDimRow();
//        dimCol = biases.getDimCol();
//        String dbiases = "[ ";
//        for (int r = 0; r < dimRow; r++) {
//            for (int c = 0; c < dimCol; c++) {
//                dbiases += biases.getData(r, c).getDerivative() + " ";
//            }
//        }
//        dbiases += " ]";
//        return
//                "Receive: " + receive.toString() + "\n"
//                        + "Weights: " + weights.toString() + "\n"
//                        + "dW: " + dweight + "\n"  + "Biases: " + biases.toString() + "\n"  + "dB: " + dbiases + "\n"
//                        + "Combine: " + combine.toString() + "\n" + "Activate: " + activate.toString() + "\n";
//
//
//    }




}
