package model;

import model.neuralnetwork.Neuron;
import model.neuralnetwork.Synapse;
import model.tensor.Matrix;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;
import model.operation.Process;

public class NeuronTest {

    @Test
    public void testReceive() throws Exception {
        Neuron N1 = new Neuron(Process::rowAppend, Process::linear, Process::sigmoid);
        Neuron N2 = new Neuron();
        Neuron N3 = new Neuron();
        N3.setReceiveFunction(Process::rowAppend);

        N1.setActivateData(Process.randMat(1, 1, "Xavier"));
        N2.setActivateData(Process.randMat(1, 1, "Zero"));
        N3.addPrev(N1);
        N3.addPrev(N2);
        N3.receive();

        Synapse s1 = N1.getActivateData().getSynapse(0, 0);
        Synapse s2 = N2.getActivateData().getSynapse(0,0);
        Synapse[][] synapses = new Synapse[][]{{s1, s2}};
        Matrix toCompare = new Matrix(synapses);
        boolean flag;
        for(int c = 0; c < 2; c++){
            flag = (N3.getReceiveData().getSynapse(0, c).getValue() == toCompare.getSynapse(0, c).getValue());
            assertTrue(flag);
        }
        assertTrue(toCompare.getDimRow() == N3.getReceiveData().getDimRow() &&
                        toCompare.getDimCol() == N3.getReceiveData().getDimCol());
    }

    @Test
    public void testCombine() throws Exception {
        Neuron N1 = new Neuron(Process::rowAppend, Process::identityCombine, Process::softmax);
        N1.setCombineFunction(Process::linear);
        Matrix randomData = Process.randMat(1, 5, "Xavier");
        Matrix randomWeights = Process.randMat(1, 5, "Xavier");
        Matrix randomBiases = Process.randMat(1, 1, "Xavier");
        N1.setReceiveData(randomData);
        N1.setWeights(randomWeights);
        N1.setBiases(randomBiases);
        Matrix toCompare = Process.dot(randomData, randomWeights, randomBiases);
        N1.combine();
        boolean flag;
        for(int r = 0; r < toCompare.getDimRow(); r++){
            for(int c = 0; c < toCompare.getDimCol(); c++){
                flag = N1.getCombineData().getSynapse(r,c).getValue() == toCompare.getSynapse(r, c).getValue();
                assertTrue(flag);
            }
        }
        assertTrue(toCompare.getDimRow() == N1.getCombineData().getDimRow() &&
                toCompare.getDimCol() == N1.getCombineData().getDimCol());

        Neuron N2 = new Neuron();
        N2.setCombineFunction(Process::identityCombine);
        N2.setReceiveData(randomData);
        N2.combine();
        assertTrue(N2.getCombineData().equals(randomData));


    }

    @Test
    public void testActivate() throws Exception {
        Neuron N1 = new Neuron();
        N1.setActivateFunction(Process::identityActivate);
        N1.setCombineData(new Matrix(new Synapse[][]{{new Synapse(2)}}));
        N1.activate();
        assertTrue(N1.getActivateData().getSynapse(0,0).getValue() == new Synapse(2).getValue());
    }

    @Test
    public void testPrev() {
        Neuron neuron = new Neuron();
        Neuron prev = new Neuron();
        neuron.addPrev(prev);
        assertTrue(neuron.getPrev().get(0).equals(prev));
    }

    @Test
    public void testAdj() {
        Neuron neuron = new Neuron();
        Neuron adj = new Neuron();
        neuron.addAdj(adj);
        assertTrue(neuron.getAdj(0).equals(adj));
    }

    @Test
    public void testNext() {
        Neuron neuron = new Neuron();
        Neuron next = new Neuron();
        neuron.addNext(next);
        assertTrue(neuron.getNext().get(0).equals(next));
    }
}
