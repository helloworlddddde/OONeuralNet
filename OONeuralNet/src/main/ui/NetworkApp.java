package ui;

import model.neuralnetwork.Network;
import model.operation.Process;
import model.tensor.Matrix;
import persistence.JsonReader;
import persistence.JsonWriter;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

// Class responsible for user interface; providing text instructions and accepting user's inputs
public class NetworkApp {

    private List<double[]> dataList;
    private List<double[]> expectedList;
    private Network network;
    private Scanner reader = new Scanner(System.in);
    private int numberOfLayers;
    private int epochs;
    private double learningRate;
    private boolean running = true;

    public NetworkApp() throws Exception {

        System.out.println("(1) load network with saved weights and biases");
        System.out.println("(2) Initialize new network");
        int choiceLoad = Integer.parseInt(reader.nextLine());

        if (choiceLoad == 1) {
            System.out.println("file location?");
            try {
                loadNetwork(reader.nextLine());
            } catch (FileNotFoundException e) {
                System.out.println("File not found, terminating program");
                return;
            }
        }

        if (choiceLoad == 2) {
            initializeNetwork();
        }

        readData();

        trainNetwork();

        while (running) {
            testNetwork();
        }









    }

    // REQUIRES: fileName != null
    // MODIFIES: this
    // EFFECTS: load network from fileLocation
    private void loadNetwork(String fileLocation) throws FileNotFoundException {

        JsonReader jsonReader = new JsonReader(fileLocation);
        try {
            network = jsonReader.read();
        } catch (IOException e) {
            throw new FileNotFoundException();
        }

    }

    private void trainNetwork() throws Exception {
        System.out.println("How many epochs to train for?");
        epochs = Integer.parseInt(reader.nextLine());
        System.out.println("Learning rate?");
        learningRate = Double.parseDouble(reader.nextLine());


        for (int i = 0; i < epochs; i++) {
            System.out.println("Epoch: " + i);
            for (int j = 0; j < dataList.size(); j++) {
                Matrix[] input = Process.listToNormalizedInput(dataList.get(j));
                Matrix[] expected = Process.listToOutput(expectedList.get(j));
                System.out.println("Loss for data " + j + ": \n"
                        + network.backProp(Process::crossEntropy, input, expected, 1));
            }
        }

    }

    private void initializeNetwork() {
        ArrayList<Integer> tempLayer = new ArrayList<Integer>();
        System.out.println("How many layers?");
        numberOfLayers = Integer.parseInt(reader.nextLine());

        for (int i = 0; i < numberOfLayers; i++) {
            System.out.println("Size of layer " + i + "?");
            tempLayer.add(Integer.parseInt(reader.nextLine()));
        }

        network = Network.multilayerPerceptron(tempLayer.stream().mapToInt(n -> n).toArray());

        initializeLayers();

    }

    private void initializeLayers() {
        for (int i = 0; i < numberOfLayers; i++) {
            System.out.println("Activation function of layer " + i + "?");
            System.out.println("(1) Hyperbolic Tangent");
            System.out.println("(2) Logistic Sigmoid");
            System.out.println("(3) Softmax");

            switch (Integer.parseInt(reader.nextLine())) {
                case 1: {
                    network.getLayer(i).setActivateFunction(Process::tanh);
                    break;
                }

                case 2: {
                    network.getLayer(i).setActivateFunction(Process::sigmoid);
                    break;
                }

                case 3: {
                    network.getLayer(i).setActivateFunction(Process::softmax);
                    break;
                }
            }
        }
    }

    private void readData() {
        String[] stringData;
        dataList = new ArrayList<double[]>();
        expectedList = new ArrayList<double[]>();

        System.out.println("How many data?");

        int num = Integer.parseInt(reader.nextLine());
        for (int i = 0; i < num; i++) {
            System.out.println(i + ": Type in your data in the form x1, x2, x3, ... , xn of length "
                    + network.getLayer(0).getSize());
            double[] tempData = new double[network.getLayer(0).getSize()];
            stringData = reader.nextLine().split(",", 0);
            for (int j = 0; j < stringData.length; j++) {
                tempData[j] = Double.parseDouble(stringData[j]);
            }
            dataList.add(tempData);
            System.out.println(i + ": Type in your expected output in the form y1, y2, y3, ..., yn of length "
                    + network.getLayer(network.getLayers().size() - 1).getSize());
            double[] tempExpected = new double[network.getLayer(0).getSize()];
            stringData = reader.nextLine().split(",", 0);
            for (int j = 0; j < stringData.length; j++) {
                tempExpected[j] = Double.parseDouble(stringData[j]);
            }
            expectedList.add(tempExpected);
        }

    }

    private void testNetwork() throws Exception {
        String[] stringData;
        System.out.println("Test your network?");
        System.out.println("Input \"save\" to save network");
        String inputString = reader.nextLine();
        if (inputString.equals("q")) {
            running = false;
            return;
        }
        if (inputString.equals("save")) {
            System.out.println("file name?");
            saveNetwork(reader.nextLine());
            return;
        }



        double[] tempData = new double[network.getLayer(0).getSize()];
        stringData = inputString.split(",", 0);
        for (int j = 0; j < stringData.length; j++) {
            tempData[j] = Double.parseDouble(stringData[j]);
        }
        System.out.println("Output: ");
        System.out.println(network.output(Process.listToOutput(tempData)));




    }

    // REQUIRES: fileName != null
    // MODIFIES: this
    // EFFECTS: save network to ./data/fileName.json
    private void saveNetwork(String fileName) throws FileNotFoundException {
        JsonWriter jsonWriter = new JsonWriter("./data/" + fileName + ".json");
        jsonWriter.open();
        jsonWriter.write(network);
        jsonWriter.close();
    }


}
