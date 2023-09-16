package model.operation;

import model.tensor.Matrix;
import model.neuralnetwork.Layer;
import model.neuralnetwork.Neuron;
import model.neuralnetwork.Synapse;

import java.util.ArrayList;

// A class with only static methods, used for running general methods on component of Network
public class Process {

    // REQUIRES: fn != null, m != null
    // MODIFIES: m
    // EFFECTS: apply fn on each Synapse in m, return the resultant matrix
    public static Matrix map(Function fn, Matrix m) throws Exception {
        Synapse[][] synapses = new Synapse[m.getDimRow()][m.getDimCol()];
        for (int r = 0; r < m.getDimRow(); r++) {
            for (int c = 0; c < m.getDimCol(); c++) {
                synapses[r][c] = fn.compute(m.getSynapse(r, c));
            }
        }
        return new Matrix(synapses);
    }

    // REQUIRES: m != null
    // MODIFIES: m
    // EFFECTS: autoDifferentiate each Synapse in m
    public static void autoDifferentiate(Matrix m) {
        for (int r = 0; r < m.getDimRow(); r++) {
            for (int c = 0; c < m.getDimCol(); c++) {
                m.getSynapse(r, c).autoDifferentiate();
            }
        }
    }

//    public static Matrix binaryCrossEntropy(Layer l, Matrix[] expected) {
//        Synapse[] result = new Synapse[l.getSize()];
//        for (int i = 0; i < l.getSize(); i++) {
//            Synapse loss = new Synapse(0);
//            Matrix nodes = l.getNeuron(i).getActivateData();
//            Matrix y = expected[i];
//            int dimRow = nodes.getDimRow();
//            int dimCol = nodes.getDimCol();
//            for (int r = 0; r < dimRow; r++) {
//                for (int c = 0; c < dimCol; c++) {
//                    loss = Synapse.plus(Synapse.multiply(y.getData(r, c), Synapse.ln(nodes.getData(r, c))),
//                            Synapse.multiply(Synapse.plus(new Synapse(1),
//                                    Synapse.multiply(new Synapse(-1), y.getData(r, c))),
//                                    Synapse.ln(Synapse.plus(new Synapse(1),
//                                            Synapse.multiply(new Synapse(-1), nodes.getData(r, c))))));
//                }
//            }
//            result[i] = Synapse.multiply(new Synapse(-1), loss);
//
//        }
//        return new Matrix(new Synapse[][]{result});
//    }

//    public static Matrix squareError(Layer l, Matrix[] expected) {
//        Synapse[] result = new Synapse[l.getSize()];
//        for (int i = 0; i < l.getSize(); i++) {
//            Synapse loss = new Synapse(0);
//            Matrix nodes = l.getNeuron(i).getActivateData();
//            Matrix y = expected[i];
//            int dimRow = nodes.getDimRow();
//            int dimCol = nodes.getDimCol();
//            for (int r = 0; r < dimRow; r++) {
//                for (int c = 0; c < dimCol; c++) {
//                    loss = Synapse.plus(Synapse.pow(Synapse.plus(y.getData(r, c),
//                            (Synapse.multiply(nodes.getData(r, c), new Synapse(-1)))), 2), loss);
//                }
//            }
//            result[i] = loss;
//
//        }
//        return new Matrix(new Synapse[][]{result});
//    }

    ;

    // REQUIRES: l != null
    // MODIFIES: nothing
    // EFFECTS: compare the activation data for neurons in l and matrices in expected to compute
    // the cross entropy loss
    public static Matrix crossEntropy(Layer l, Matrix[] expected) {
        Synapse sum = new Synapse(0);
        for (int i = 0; i < l.getSize(); i++) {
            Matrix nodes = l.getNeuron(i).getActivateData();
            Matrix y = expected[i];
            int dimRow = nodes.getDimRow();
            int dimCol = nodes.getDimCol();
            for (int r = 0; r < dimRow; r++) {
                for (int c = 0; c < dimCol; c++) {
                    sum = Synapse.plus(Synapse.multiply(y.getSynapse(r, c), Synapse.ln(nodes.getSynapse(r, c))), sum);
                }
            }
        }
        sum = Synapse.multiply(new Synapse(-1), sum);
        return new Matrix(new Synapse[][]{{sum}});
    }

    // REQUIRES: neuron != null
    // MODIFIES: neuron
    // EFFECTS: set combination data of neuron to its receive data
    public static void identityCombine(Neuron neuron) {
        neuron.setCombineData(neuron.getReceiveData());
    }

    // REQUIRES: neuron != null
    // MODIFIES: neuron
    // EFFECTS: set activation data of neuron to its combination data
    public static void identityActivate(Neuron neuron) {
        neuron.setActivateData(neuron.getCombineData());
    }

    // REQUIRES: n != null
    // MODIFIES: n
    // EFFECTS: run the Softmax activation function on neuron n
    public static void softmax(Neuron n) {

        Matrix nodes = n.getCombineData();
        int dimRow = nodes.getDimRow();
        int dimCol = nodes.getDimCol();
        Synapse sum = new Synapse(0);
        for (int r = 0; r < dimRow; r++) {
            for (int c = 0; c < dimCol; c++) {
                sum = Synapse.plus(Synapse.exp(nodes.getSynapse(r, c)), sum);
            }
        }

        sum = softmaxSumAdj(n, sum);


        nodes = n.getCombineData();
        dimRow = nodes.getDimRow();
        dimCol = nodes.getDimCol();
        Synapse[][] tempSynapses = new Synapse[dimRow][dimCol];
        for (int r = 0; r < dimRow; r++) {
            for (int c = 0; c < dimCol; c++) {
                Synapse tempSynapse = n.getCombineData().getSynapse(r, c);
                tempSynapses[r][c] = Synapse.multiply(Synapse.exp(tempSynapse), Synapse.pow(sum, -1));
            }
        }
        n.setActivateData(new Matrix(tempSynapses));

    }

    // REQUIRES: n != null
    // MODIFIES: n
    // EFFECTS: sum the softmax component of adjacent neurons of n
    public static Synapse softmaxSumAdj(Neuron n, Synapse sum) {
        for (int i = 0; i < n.getAdj().size(); i++) {
            Matrix nodes = n.getAdj(i).getCombineData();
            int dimRow = nodes.getDimRow();
            int dimCol = nodes.getDimCol();
            for (int r = 0; r < dimRow; r++) {
                for (int c = 0; c < dimCol; c++) {
                    sum = Synapse.plus(Synapse.exp(nodes.getSynapse(r, c)), sum);
                }
            }
        }
        return sum;
    }

//    public static void softplus(Neuron neuron) throws Exception {
//        Matrix nodes = neuron.getCombineData();
//        Matrix result = map(Process::softplus, nodes);
//        neuron.setActivateData(result);
//    }
//
//    public static Synapse softplus(Synapse synapse) {
//        Synapse resultSynapse = Synapse.ln(Synapse.plus(new Synapse(1), Synapse.exp(synapse)));
//        return resultSynapse;
//    }

    // REQUIRES: neuron != null
    // MODIFIES: neuron
    // EFFECTS: run the hyperbolic tangent function on the combination data of neuron, and set the resultant data to
    // its activation data
    public static void tanh(Neuron neuron) throws Exception {
        Matrix nodes = neuron.getCombineData();
        Matrix result = map(Process::tanh, nodes);
        neuron.setActivateData(result);
    }

    // REQUIRES: synapse != null
    // MODIFIES: synapse
    // EFFECTS: run the hyperbolic tangent function on the synapse and return the result
    public static Synapse tanh(Synapse synapse) {
        Synapse resultSynapse = Synapse.multiply(Synapse.plus(
                Synapse.exp(Synapse.multiply(synapse, new Synapse(2))), new Synapse(-1)),
                Synapse.pow(Synapse.plus(Synapse.exp(Synapse.multiply(synapse, new Synapse(2))), new Synapse(1)), -1));
        return resultSynapse;
    }


    // REQUIRES: neuron != null
    // MODIFIES: neuron
    // EFFECTS: run the logistic sigmoid function on the combination data of neuron, and set the resultant data to
    // its activation data
    public static void sigmoid(Neuron neuron) throws Exception {
        Matrix nodes = neuron.getCombineData();
        Matrix result = map(Process::sigmoid, nodes);
        neuron.setActivateData(result);
    }


    // REQUIRES: synapse != null
    // MODIFIES: synapse
    // EFFECTS: run the logistic sigmoid function on the synapse and return the result
    public static Synapse sigmoid(Synapse synapse) {
        Synapse resultSynapse = Synapse.pow(Synapse.plus(new Synapse(1),
                Synapse.exp(Synapse.multiply(new Synapse(-1), synapse))), -1);
        return resultSynapse;
    }

    // REQUIRES: neuron != null
    // MODIFIES: neuron
    // EFFECTS: take the linear combination of the neuron's receive data, weights, and biases, then set the combination
    // then set the result as the neuron's combination data
    public static void linear(Neuron neuron) {
        Matrix nodes = neuron.getReceiveData();
        Matrix weights = neuron.getWeights();
        Matrix bias = neuron.getBiases();
        Matrix result = dot(nodes, weights, bias);
        neuron.setCombineData(result);
    }

    // REQUIRES: neuron != null
    // MODIFIES: neuron
    // EFFECTS: combine the matrices of previous neurons by appending them in a 1 x N matrix, then set the result
    // as the neuron's receive data
    public static void rowAppend(Neuron neuron) {
        ArrayList<Neuron> prev = neuron.getPrev();
        ArrayList<Synapse> synapses = new ArrayList<Synapse>();
        for (int i = 0; i < prev.size(); i++) {
            Matrix temp = flatten(prev.get(i).getActivateData());
            for (int c = 0; c < temp.getDimCol(); c++) {
                synapses.add(temp.getSynapse(0, c));
            }
        }
        Synapse sum = new Synapse(0);
        Synapse[][] result = new Synapse[1][synapses.size()];
        for (int c = 0; c < synapses.size(); c++) {
            result[0][c] = synapses.get(c);

        }

        neuron.setReceiveData(new Matrix(result));
    }

    // REQUIRES: row >= 1, col >= 1, mode != null
    // MODIFIES: nothing
    // EFFECTS: return a row x col matrix with the specified mode of initialization
    public static Matrix randMat(int row, int col, String mode) {
        Synapse[][] synapses = new Synapse[row][col];
        switch (mode) {
            case "Xavier":
                return xavierMatrix(row, col);
            case "Zero":
                return zeroMatrix(row, col);
            case "All-ones":
                return oneMatrix(row, col);
            default:
                return zeroMatrix(row, col);
        }
    }

    // REQUIRES: row >= 1, col >= 1
    // MODIFIES: nothing
    // EFFECTS: return a xavier-initialized row x col matrix
    public static Matrix xavierMatrix(int row, int col) {
        Synapse[][] synapses = new Synapse[row][col];
        for (int r = 0; r < row; r++) {
            for (int c = 0; c < col; c++) {
                synapses[r][c] = new Synapse(2 * (Math.random() - 0.5) * (1 / Math.sqrt(row * col)));
            }
        }
        return new Matrix(synapses);
    }

    // REQUIRES: row >= 1, col >= 1
    // MODIFIES: nothing
    // EFFECTS: return a row x col zero matrix
    public static Matrix zeroMatrix(int row, int col) {
        Synapse[][] synapses = new Synapse[row][col];
        for (int r = 0; r < row; r++) {
            for (int c = 0; c < col; c++) {
                synapses[r][c] = new Synapse(0);
            }
        }
        return new Matrix(synapses);
    }

    // REQUIRES: row >= 1, col >= 1
    // MODIFIES: nothing
    // EFFECTS: return a row x col all-ones matrix
    public static Matrix oneMatrix(int row, int col) {
        Synapse[][] synapses = new Synapse[row][col];
        for (int r = 0; r < row; r++) {
            for (int c = 0; c < col; c++) {
                synapses[r][c] = new Synapse(1);
            }
        }
        return new Matrix(synapses);
    }


    // REQUIRES: m != null
    // MODIFIES: nothing
    // EFFECTS: take a row x col matrix m and return a 1 x (row + col) matrix with the elements of m
    // in row-major order
    public static Matrix flatten(Matrix m) {
        Synapse[][] synapses = new Synapse[1][m.getDimRow() * m.getDimCol()];
        for (int r = 0; r < m.getDimRow(); r++) {
            for (int c = 0; c < m.getDimCol(); c++) {
                synapses[0][r + c] = m.getSynapse(r, c);

            }
        }


        return new Matrix(synapses);
    }

    // REQUIRES: m1 != null, m2 != null, m3 != null, m1 and m2 has the same dimensions, m3 has dimension 1 x 1,
    // MODIFIES: m1, m2, m3
    // EFFECTS: return (m1 * m2) + m3, where * is the dot product (summation of element-wise
    // multiplication of m1 and m2)
    public static Matrix dot(Matrix m1, Matrix m2, Matrix m3) {
        Synapse sum = new Synapse(0);
        for (int r = 0; r < m1.getDimRow(); r++) {
            for (int c = 0; c < m1.getDimCol(); c++) {
                sum = Synapse.plus(Synapse.multiply(m1.getSynapse(r, c), m2.getSynapse(r, c)), sum);
            }
        }

        return new Matrix(new Synapse[][]{{Synapse.plus(sum, m3.getSynapse(0, 0))}});
    }

    // REQUIRES: nothing
    // MODIFIES: nothing
    // EFFECTS: return a 1d array of 1 x 1 matrices with a normalized element in the list of input data
    public static Matrix[] listToNormalizedInput(double... listOfData) {
        Matrix[] m = new Matrix[listOfData.length];
        double sum = 0;
        for (int i = 0; i < listOfData.length; i++) {
            sum += Math.pow(listOfData[i], 2);
        }
        if (sum == 0) {
            sum = 1;
        }
        sum = Math.sqrt(sum);
        for (int i = 0; i < listOfData.length; i++) {
            Synapse data = new Synapse(listOfData[i] / sum);
            m[i] = new Matrix(new Synapse[][]{{data}});
        }
        return m;
    }

    // REQUIRES: nothing
    // MODIFIES: nothing
    // EFFECTS: return a 1d array of 1 x 1 matrices with an element in the list of input data
    public static Matrix[] listToOutput(double... listOfData) {
        Matrix[] m = new Matrix[listOfData.length];
        for (int i = 0; i < listOfData.length; i++) {
            Synapse data = new Synapse(listOfData[i]);
            m[i] = new Matrix(new Synapse[][]{{data}});
        }
        return m;
    }


}
