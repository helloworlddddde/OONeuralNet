package model;

import model.neuralnetwork.Layer;
import model.neuralnetwork.Neuron;
import model.neuralnetwork.Synapse;
import model.tensor.Matrix;
import org.junit.jupiter.api.Test;
import model.operation.Process;
import static org.junit.jupiter.api.Assertions.*;

public class LayerTest {

    @Test
    public void testGetSize() {
        Layer layer = new Layer(314, Process::rowAppend, Process::linear, Process::tanh);
        assertEquals(layer.getSize(), 314);
    }

    @Test
    public void testGetNeuron() {
        Layer layer = new Layer(5, Process::rowAppend, Process::linear, Process::tanh);
        Neuron neuron = new Neuron();
        layer.addNeuron(neuron);
        assertTrue(layer.getNeuron(5).equals(neuron));
    }

    @Test
    public void testAddNeuron() {
        Layer layer = new Layer(5, Process::rowAppend, Process::linear, Process::tanh);
        assertTrue(layer.getSize() == 5);
        Neuron neuron = new Neuron(Process:: rowAppend, Process::identityCombine, Process::softmax);
        layer.addNeuron(neuron);
        assertTrue(layer.getSize() == 6);
        assertTrue(layer.getNeurons().contains(neuron));

    }

    @Test
    public void testReceiveMethods() throws Exception {
        Layer testLayer1 = new Layer(1, Process::rowAppend, Process::linear, Process::tanh);
        Layer testLayer2 = new Layer(1, Process::rowAppend, Process::linear, Process::tanh);
        testLayer2.setReceiveFunction(Process::rowAppend);
        testLayer2.getNeuron(0).addPrev(testLayer1.getNeuron(0));
        Synapse testSynapse = new Synapse(1);
        testLayer1.getNeuron(0).setActivateData(new Matrix(new Synapse[][]{{testSynapse}}));
        testLayer2.receive();
        assertTrue(testLayer2.getNeuron(0).getReceiveData().getSynapse(0, 0).equals(
                testLayer1.getNeuron(0).getActivateData().getSynapse(0,0)));
    }

    @Test
    public void testCombineMethods() throws Exception {
        Layer testLayer = new Layer(1, Process::rowAppend, Process::linear, Process::tanh);
        testLayer.setCombineFunction(Process::identityCombine);
        Synapse testSynapse = new Synapse(2);
        testLayer.getNeuron(0).setReceiveData(new Matrix(new Synapse[][]{{testSynapse}}));
        testLayer.combine();
        assertTrue(testLayer.getNeuron(0).getCombineData().getSynapse(0, 0).equals(
                testLayer.getNeuron(0).getReceiveData().getSynapse(0, 0)));
    }

    @Test
    public void testActivateMethods() throws Exception {
        Layer testLayer = new Layer(1, Process::rowAppend, Process::linear, Process::tanh);
        testLayer.setActivateFunction(Process::identityActivate);
        Synapse testSynapse = new Synapse(3);
        testLayer.getNeuron(0).setCombineData(new Matrix(new Synapse[][]{{testSynapse}}));
        testLayer.activate();
        assertTrue(testLayer.getNeuron(0).getActivateData().getSynapse(0, 0).equals(
                testLayer.getNeuron(0).getCombineData().getSynapse(0, 0)));
    }

    @Test
    public void testFireAll() throws Exception {
        Layer testLayer1 = new Layer(1, Process::rowAppend, Process::linear, Process::tanh);
        Layer testLayer2 = new Layer(1, Process::rowAppend, Process::linear, Process::tanh);
        testLayer2.setReceiveFunction(Process::rowAppend);
        testLayer2.setCombineFunction(Process::identityCombine);
        testLayer2.setActivateFunction(Process::identityActivate);
        testLayer2.getNeuron(0).addPrev(testLayer1.getNeuron(0));
        Synapse testSynapse = new Synapse(1);
        testLayer1.getNeuron(0).setActivateData(new Matrix(new Synapse[][]{{testSynapse}}));
        testLayer2.fireAll();
        assertTrue(testLayer2.getNeuron(0).getReceiveData().getSynapse(0, 0).equals(
                testLayer1.getNeuron(0).getActivateData().getSynapse(0,0)));
        assertTrue(testLayer2.getNeuron(0).getCombineData().getSynapse(0, 0).equals(
                testLayer2.getNeuron(0).getReceiveData().getSynapse(0, 0)));
        assertTrue(testLayer2.getNeuron(0).getActivateData().getSynapse(0, 0).equals(
                testLayer2.getNeuron(0).getCombineData().getSynapse(0, 0)));
    }

    @Test
    public void testGradientDescent() {
        Layer testLayer = new Layer(1, Process::rowAppend, Process::linear, Process::tanh);
        testLayer.getNeuron(0).setWeights(new Matrix(new Synapse[][]{{new Synapse(6)}}));
        testLayer.getNeuron(0).setBiases(new Matrix(new Synapse[][]{{new Synapse(7)}}));
        Synapse x = testLayer.getNeuron(0).getWeights().getSynapse(0, 0);
        Synapse y = testLayer.getNeuron(0).getBiases().getSynapse(0, 0);
        Synapse f = Synapse.multiply(
                Synapse.exp(x),
                Synapse.pow(Synapse.ln(y), -1)
        );

        f.autoDifferentiate();

        double gradX = x.getDerivative();
        double gradY = y.getDerivative();
        testLayer.gradientDescent(0.5);
        assertTrue(x.getValue() == 6 - 0.5 * gradX);
        assertTrue(y.getValue() == 7 - 0.5 * gradY);

        f = Synapse.multiply(
                Synapse.exp(x),
                Synapse.pow(Synapse.ln(y), -1)
        );
        double tempValue = x.getValue();
        Synapse z = Synapse.multiply(new Synapse(2), x);
        z.autoDifferentiate();
        testLayer.gradientDescent(1);
        assertTrue(x.getValue() == tempValue - 2);
    }



}
