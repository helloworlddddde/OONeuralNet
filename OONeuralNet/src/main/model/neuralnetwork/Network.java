package model.neuralnetwork;

import model.Event;
import model.EventLog;
import model.tensor.Matrix;
import model.operation.Error;
import model.operation.Process;
import org.json.JSONArray;
import org.json.JSONObject;
import persistence.Writable;

import java.util.ArrayList;
import java.util.Iterator;

// A Network is a class which includes the list of layers belonging to it as well as basic methods for the layers.
// A Network can deals with network initialization and Backpropagation.
public class Network implements Writable {

    //<editor-fold desc="Fields of Network">
    private ArrayList<Layer> layers = new ArrayList<Layer>();
    //</editor-fold>

    //<editor-fold desc="Network Constructors">

    // REQUIRES: nothing
    // MODIFIES this;
    // EFFECTS: create an empty Network with empty layers
    public Network() {
    }
    //</editor-fold>

    // REQUIRES: nothing
    // MODIFIES: layers
    // EFFECTS: fully connect all neurons in the layers
    public void fullConnect() {
        for (Layer layer : layers) {
            for (Neuron neuron : layer.getNeurons()) {
                neuron.getPrev().clear();
                neuron.getNext().clear();
            }
        }
        for (int i = 0; i < layers.size() - 1; i++) {
            Layer l1 = layers.get(i);
            Layer l2 = layers.get(i + 1);

            for (int m = 0; m < l2.getSize(); m++) {
                Neuron n2 = l2.getNeuron(m);
                for (int n = 0; n < l1.getSize(); n++) {
                    Neuron n1 = l1.getNeuron(n);
                    n2.addPrev(n1);
                    n1.addNext(n2);
                }
                n2.setWeights(Process.randMat(1, l1.getSize(), "Xavier"));
                n2.setBiases(Process.randMat(1, 1, "Xavier"));
            }
        }
        EventLog.getInstance().logEvent(new Event("Parameters reset with neurons reconnection"));
    }

    // REQUIRES: nothing
    // MODIFIES: nothing
    // EFFECTS: log the data input event to console
    public static void logInput(String inputText) {
        EventLog.getInstance().logEvent(new Event("Input to Network: " + inputText));
    }

    // REQUIRES: nothing
    // MODIFIES: nothing
    // EFFECTS: print to console all previously logged events from when the application last started
    public static void endNetwork() {
        Iterator<Event> eventIterator = EventLog.getInstance().iterator();
        while (eventIterator.hasNext()) {
            Event currentEvent = eventIterator.next();
            System.out.println(currentEvent.getDate() + " " + currentEvent.getDescription());
        }
        EventLog.getInstance().clear();
    }

    // REQUIRES: nothing
    // MODIFIES: layers
    // EFFECTS: fully connect all neurons in the layers without randomizing weights and biases
    public void fullConnectSameParameters() {
        for (Layer layer : layers) {
            for (Neuron neuron : layer.getNeurons()) {
                neuron.getPrev().clear();
                neuron.getNext().clear();
            }
        }
        for (int i = 0; i < layers.size() - 1; i++) {
            Layer l1 = layers.get(i);
            Layer l2 = layers.get(i + 1);

            for (int m = 0; m < l2.getSize(); m++) {
                Neuron n2 = l2.getNeuron(m);
                for (int n = 0; n < l1.getSize(); n++) {
                    Neuron n1 = l1.getNeuron(n);
                    n2.addPrev(n1);
                    n1.addNext(n2);
                }
            }
        }
    }


    // REQUIRES: nothing
    // MODIFIES: layers
    // EFFECTS: run the gradient descent algorithm on each layer in layers
    public void gradientDescent(double learningRate) {
        for (int i = 1; i < layers.size(); i++) {
            layers.get(i).gradientDescent(learningRate);
        }
    }

    // REQUIRES: input != null
    // MODIFIES: layers
    // EFFECTS: set the activation data for neurons in the first layer to be input, then fire all neurons
    // in every other layer
    public void fire(Matrix[] input) throws Exception {
        for (int i = 0; i < layers.get(0).getSize(); i++) {
            layers.get(0).getNeuron(i).setActivateData(input[i]);
        }
        for (int i = 1; i < layers.size(); i++) {
            layers.get(i).fireAll();
        }
    }

    // REQUIRES: lossFn != null, input != null, expected != null
    // MODIFIES: layers
    // EFFECTS: fire all neurons in layers, autoDifferentiate the resultant from running lossFn, then run gradient
    // descent on all the layers.
    public Matrix backProp(Error lossFn, Matrix[] input, Matrix[] expected, double learningRate) throws Exception {
        fire(input);
        Matrix loss = lossFn.compute(layers.get(layers.size() - 1), expected);
        Process.autoDifferentiate(loss);
        String outputText = layers.get(layers.size() - 1).toString();
        outputText += "Loss: " + loss;
        //System.out.println(outputText);
        gradientDescent(learningRate);
        return loss;
    }


    // REQUIRES: input != null
    // MODIFIES: this
    // EFFECTS: feed forward the input data nad output the text of the result
    public String output(Matrix[] input) throws Exception {
        fire(input);
        String output = "";
        Layer l = layers.get(layers.size() - 1);
        for (int i = 0; i < l.getSize(); i++) {
            output += l.getNeuron(i).getActivateData();
        }
        EventLog.getInstance().logEvent(new Event("Output of Network: " + "\n" + output));
        return output;
    }

    // REQUIRES: layer != null, 0 <= i <= layers.size()
    // MODIFIES: layers
    // MODIFIES: add layer to layers
    public void addLayer(Layer layer, int i) {
        layers.add(i, layer);
        EventLog.getInstance().logEvent(new Event("Added layer of size " + layer.getSize()
                + " to network"));
    }

    // REQUIRES: layer != null
    // MODIFIES: layers
    // MODIFIES: add layer to layers
    public void addLayer(Layer layer) {
        layers.add(layer);
        EventLog.getInstance().logEvent(new Event("Added layer of size " + layer.getSize()
                + " to network"));
    }

//    public void addLayers(Layer... layers) {
//        for (Layer layer : layers) {
//            this.layers.add(layer);
//        }
//    }


    // REQUIRES: i is an integer between 0 to layers.size() - 1
    // MODIFIES: this
    // EFFECTS: return the ith layer in layers
    public Layer getLayer(int i) {
        return layers.get(i);
    }

    // REQUIRES: all integers in sizes must be 1 or greater
    // MODIFIES: layers
    // REQUIRES: create a multilayer perceptron with the corresponding size of each layer
    public static Network multilayerPerceptron(int... sizes) {
        Network net = new Network();
        for (int i = 0; i < sizes.length - 1; i++) {
            net.addLayer(new Layer(sizes[i], Process::rowAppend, Process::linear, Process::tanh));
        }
        net.addLayer(new Layer(sizes[sizes.length - 1], Process::rowAppend, Process::linear, Process::softmax));
        net.fullConnect();
        return net;
    }

    public ArrayList<Layer> getLayers() {
        return layers;
    }



    // Specifications and general structure are based on https://github.students.cs.ubc.ca/CPSC210/JsonSerializationDemo
    // EFFECTS: convert Network to a JSONObject, storing the appropriate corresponding data
    @Override
    public JSONObject toJson() {
        JSONObject json = new JSONObject();
        JSONArray jsonArray = new JSONArray();
        for (int k = 0; k < layers.size(); k++) {
            Layer layer = layers.get(k);
            JSONObject tempLayer = new JSONObject();
            JSONArray jsonArray2 = new JSONArray();
            tempLayer.put("size", layer.getSize());
            jsonArray.put(tempLayer);
            for (Neuron neuron : layer.getNeurons()) {
                JSONObject tempNeuron = new JSONObject();
//                tempNeuron.put("Receive function", neuron.getReceiveFunction());
//                tempNeuron.put("Combine function", neuron.getCombineFunction());
//                tempNeuron.put("Activate function", neuron.getActivateFunction());
                if (neuron.getWeights() == null) {
                    tempNeuron.put("Weights", "");
                    tempNeuron.put("Biases", "");
                } else {
                    tempNeuron.put("Weights", neuron.getWeights().toString2());
                    tempNeuron.put("Biases", neuron.getBiases().toString2());
                }

                jsonArray2.put(tempNeuron);
            }
            tempLayer.put("Neurons", jsonArray2);


        }
        json.put("Layers", jsonArray);

        return json;
    }


}
