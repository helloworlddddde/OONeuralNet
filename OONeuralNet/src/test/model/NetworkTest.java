package model;

import model.neuralnetwork.Layer;
import model.neuralnetwork.Network;
import model.neuralnetwork.Neuron;
import model.neuralnetwork.Synapse;
import model.operation.Process;
import model.tensor.Matrix;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class NetworkTest {

    @Test
    public void testMultilayerPerceptron() {
        Network net = Network.multilayerPerceptron(2, 4, 9, 16);
        net.fullConnectSameParameters();
        Layer layer0 = net.getLayer(0);
        assertTrue(layer0.getSize() == 2);
        for (Neuron neuron : layer0.getNeurons()) {
            assertTrue(neuron.getPrev().size() == 0);
            assertTrue(neuron.getAdj().size() == 1);
            assertTrue(neuron.getNext().size() == 4);
        }
        Layer layer1 = net.getLayer(1);
        assertTrue(layer1.getSize() == 4);
        for (Neuron neuron : layer1.getNeurons()) {
            assertTrue(neuron.getPrev().size() == 2);
            assertTrue(neuron.getAdj().size() == 3);
            assertTrue(neuron.getNext().size() == 9);
        }
        Layer layer2 = net.getLayer(2);
        assertTrue(layer2.getSize() == 9);
        for (Neuron neuron : layer2.getNeurons()) {
            assertTrue(neuron.getPrev().size() == 4);
            assertTrue(neuron.getAdj().size() == 8);
            assertTrue(neuron.getNext().size() == 16);
        }
        Layer layer3 = net.getLayer(3);
        assertTrue(layer3.getSize() == 16);
        for (Neuron neuron : layer3.getNeurons()) {
            assertTrue(neuron.getPrev().size() == 9);
            assertTrue(neuron.getAdj().size() == 15);
            assertTrue(neuron.getNext().size() == 0);
        }
    }

    @Test
    public void testBackProp() throws Exception {
        // loss values should strictly decrease at least for the first few iterations
        Network test = Network.multilayerPerceptron(4, 10, 4);
        Matrix[] losses1 = new Matrix[5];
        Matrix[] losses2 = new Matrix[5];
        test.getLayer(1).setReceiveFunction(Process::rowAppend);
        test.getLayer(1).setCombineFunction(Process::linear);
        test.getLayer(1).setActivateFunction(Process::sigmoid);
        for (Neuron neuron : test.getLayer(1).getNeurons()) {
            neuron.setWeights(Process.randMat(1, 4, "All-ones"));
            neuron.setBiases(Process.randMat(1, 1, "All-ones"));
        }
        for (Neuron neuron : test.getLayer(2).getNeurons()) {
            neuron.setWeights(Process.randMat(1, 10, "All-ones"));
            neuron.setBiases(Process.randMat(1, 1, "All-ones"));
        }
        Matrix loss1;
        Matrix loss2;
        for (int i = 0; i < 5; i++) {
            Matrix[] input1 = Process.listToNormalizedInput(5, 5, 5, 5);
            Matrix[] input2 = Process.listToNormalizedInput(-1,-1,-1,-1);
            Matrix[] y = Process.listToOutput(1, 0, 0, 0);
            Matrix[] x = Process.listToOutput(0, 0, 0, 1);
            loss1 = test.backProp(Process::crossEntropy, input1, y, 0.1);
            loss2 = test.backProp(Process::crossEntropy, input2, x, 0.1);
            losses1[i] = loss1;
            losses2[i] = loss2;
        }
        boolean flag;
        for (int j = 0; j < 4; j++) {
            flag = losses1[j+1].getSynapse(0,0).getValue() < losses1[j].getSynapse(0,0).getValue();
            assertTrue(flag);

        }
        for (int k = 0; k < 4; k++) {
            flag = losses2[k+1].getSynapse(0,0).getValue() < losses2[k].getSynapse(0,0).getValue();
            assertTrue(flag);
        }


    }

    @Test
    public void testOutput() throws Exception {
        Network test = Network.multilayerPerceptron(4, 7, 4);



        for (int i = 0; i < 1000; i++) {
            Matrix[] input1 = Process.listToNormalizedInput(5, 5, 5, 5);
            Matrix[] input2 = Process.listToNormalizedInput(-1,-1,-1,-1);
            Matrix[] y = Process.listToOutput(1, 0, 0, 0);
            Matrix[] x = Process.listToOutput(0, 0, 0, 1);
            test.backProp(Process::crossEntropy, input1, y, 1);
            test.backProp(Process::crossEntropy, input2, x, 1);
        }
        Matrix[] input2 = Process.listToNormalizedInput(-1,-1,-1,-1);

        String actualOutput = test.output(input2);
        String line0 = test.getLayer(test.getLayers().size()-1).getNeuron(0).getActivateData().toString();
        String line1 = test.getLayer(test.getLayers().size()-1).getNeuron(1).getActivateData().toString();
        String line2 = test.getLayer(test.getLayers().size()-1).getNeuron(2).getActivateData().toString();
        String line3 = test.getLayer(test.getLayers().size()-1).getNeuron(3).getActivateData().toString();
        String expectedOutput = line0 + line1 + line2 + line3;
        assertTrue(test.output(input2).equals(expectedOutput));

    }

    @Test
    public void testGetLayer() {
        Layer testLayer = new Layer(1, Process::rowAppend, Process::sigmoid, Process::tanh);
        Network network = new Network();
        network.getLayers().add(testLayer);
        assertTrue(network.getLayer(0).equals(testLayer));
    }

    @Test
    public void testAddLayer() {
        Layer testLayer = new Layer(1, Process::rowAppend, Process::sigmoid, Process::tanh);
        Network network = new Network();
        network.addLayer(testLayer, testLayer.getSize() - 1);
        assertTrue(network.getLayers().get(0).equals(testLayer));
    }

    @Test
    public void testGradientDescent() {
        Network network = new Network();
        network.addLayer(new Layer(1, Process::rowAppend, Process::linear, Process::tanh));
        network.addLayer(new Layer(1, Process::rowAppend, Process::linear, Process::tanh));
        Layer testLayer1 = network.getLayer(0);
        Layer testLayer2 = network.getLayer(1);
        testLayer1.getNeuron(0).setWeights(new Matrix(new Synapse[][]{{new Synapse(6)}}));
        testLayer1.getNeuron(0).setBiases(new Matrix(new Synapse[][]{{new Synapse(7)}}));
        testLayer2.getNeuron(0).setWeights(new Matrix(new Synapse[][]{{new Synapse(6)}}));
        testLayer2.getNeuron(0).setBiases(new Matrix(new Synapse[][]{{new Synapse(7)}}));
        Synapse x = testLayer2.getNeuron(0).getWeights().getSynapse(0, 0);
        Synapse y = testLayer2.getNeuron(0).getBiases().getSynapse(0, 0);
        Synapse f = Synapse.multiply(
                Synapse.exp(x),
                Synapse.pow(Synapse.ln(y), -1)
        );

        f.autoDifferentiate();

        double tempWeightValue = testLayer1.getNeuron(0).getWeights().getSynapse(0,0).getValue();
        double tempBiasValue = testLayer1.getNeuron(0).getBiases().getSynapse(0, 0).getValue();
        double gradX = x.getDerivative();
        double gradY = y.getDerivative();
        network.gradientDescent(0.5);
        assertTrue(x.getValue() == 6 - 0.5 * gradX);
        assertTrue(y.getValue() == 7 - 0.5 * gradY);
        assertTrue(tempWeightValue == testLayer1.getNeuron(0).getWeights().getSynapse(0,0).getValue());
        assertTrue(tempBiasValue == testLayer1.getNeuron(0).getBiases().getSynapse(0,0).getValue());
    }

    @Test
    public void testFire() throws Exception {
        Matrix[] testInput = Process.listToNormalizedInput(1);
        Network network = Network.multilayerPerceptron(1, 1);
        Layer testLayer1 = network.getLayer(0);
        Layer testLayer2 = network.getLayer(1);
        testLayer2.setCombineFunction(Process::identityCombine);
        testLayer2.setActivateFunction(Process::identityActivate);
        network.fire(testInput);
        assertTrue(testLayer1.getNeuron(0).getActivateData().equals(testInput[0]));
        assertTrue(testLayer2.getNeuron(0).getReceiveData().getSynapse(0, 0).equals(
                testLayer1.getNeuron(0).getActivateData().getSynapse(0,0)));
        assertTrue(testLayer2.getNeuron(0).getCombineData().getSynapse(0, 0).equals(
                testLayer2.getNeuron(0).getReceiveData().getSynapse(0, 0)));
        assertTrue(testLayer2.getNeuron(0).getActivateData().getSynapse(0, 0).equals(
                testLayer2.getNeuron(0).getCombineData().getSynapse(0, 0)));
        assertTrue(testLayer2.getNeuron(0).getActivateData().getSynapse(0,0).equals(
                testInput[0].getSynapse(0,0)));
    }


}
