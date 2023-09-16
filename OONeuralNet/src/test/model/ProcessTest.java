package model;

import model.neuralnetwork.Layer;
import model.neuralnetwork.Neuron;
import model.neuralnetwork.Synapse;
import model.tensor.Matrix;
import org.junit.jupiter.api.Test;
import model.operation.Process;
import static org.junit.jupiter.api.Assertions.*;

public class ProcessTest {

    @Test
    public void testConstructor() {
        Process testProcess = new Process();
        assertTrue(testProcess instanceof Process);
    }

    @Test
    public void testMap() throws Exception {

        Matrix testMatrix = Process.randMat(5, 5, "Xavier");
        Matrix mappedMatrix = Process.map(
                (synapse) -> {return Synapse.multiply(new Synapse(2), synapse);}, //multiply all elements by 2
                testMatrix);

        boolean flag;
        for (int r = 0; r < testMatrix.getDimRow(); r++) {
            for (int c = 0; c < testMatrix.getDimCol(); c++) {
                flag = 2 * testMatrix.getSynapse(r, c).getValue() == mappedMatrix.getSynapse(r, c).getValue();
                assertTrue(flag);
            }
        }

    }

    @Test
    public void testCrossEntropy() {
        Layer testLayer = new Layer(4, Process::rowAppend, Process::linear, Process::softmax);
        int count = 1;
        for (Neuron neuron : testLayer.getNeurons()) {
            neuron.setActivateData(new Matrix(new Synapse[][]{{new Synapse(0.1*count)}}));
            count++;
        }
        Matrix[] expected = new Matrix[4];
        expected[0] = new Matrix(new Synapse[][]{{new Synapse(1)}});
        expected[1] = new Matrix(new Synapse[][]{{new Synapse(2)}});
        expected[2] = new Matrix(new Synapse[][]{{new Synapse(3)}});
        expected[3] = new Matrix(new Synapse[][]{{new Synapse(4)}});

        // crossEntropy = y1ln(x1) + y2ln(x2) + y3ln(x3) + y4ln(x4), where [y1], [y2], [y3], [y4] is the expected output
        // and [x1], [x2], [x3], [x4] is the activated output of testLayer
        // so in this case, crossEntropy = -(1ln(0.1) + 2ln(0.2) + 3ln(0.3) + 4ln(0.4))
        // which is about 12.7985 (represented as a 1 x 1 Matrix)

        double epsilon = 0.0001;
        assertTrue(
                Math.abs(Process.crossEntropy(testLayer, expected).getSynapse(0, 0).getValue() - 12.7985) < epsilon
                && Process.crossEntropy(testLayer, expected).getDimRow() == 1
                && Process.crossEntropy(testLayer, expected).getDimCol() == 1);
    }

    @Test
    public void testSoftmax() throws Exception {
        Layer testLayer = new Layer(3, Process::rowAppend, Process::linear, Process::softmax);
        int count = 1;

        for (int i = 0; i < testLayer.getSize(); i++) {
            testLayer.getNeuron(i).setCombineData(new Matrix(new Synapse[][]{{new Synapse(0.1*count)}}));
            count++;
        }

        testLayer.activate();
        double epsilon = 0.0001;

        assertTrue(
                Math.abs(testLayer.getNeuron(0).getActivateData().getSynapse(0, 0).getValue() - 0.3006096) < epsilon
                && testLayer.getNeuron(0).getActivateData().getDimRow() == 1
                && testLayer.getNeuron(0).getActivateData().getDimCol() == 1);
        assertTrue(
                Math.abs(testLayer.getNeuron(1).getActivateData().getSynapse(0, 0).getValue() - 0.332225) < epsilon
                        && testLayer.getNeuron(1).getActivateData().getDimRow() == 1
                        && testLayer.getNeuron(1).getActivateData().getDimCol() == 1);
        assertTrue(
                Math.abs(testLayer.getNeuron(2).getActivateData().getSynapse(0, 0).getValue() - 0.367165401) < epsilon
                        && testLayer.getNeuron(2).getActivateData().getDimRow() == 1
                        && testLayer.getNeuron(2).getActivateData().getDimCol() == 1);



    }

    @Test
    public void testRandMat() {
        Matrix testMatrix1 = Process.randMat(5, 5, "helloowoworwaoefwfe");
        boolean flag;
        for (int r = 0; r < testMatrix1.getDimRow(); r++) {
            for (int c = 0; c < testMatrix1.getDimCol(); c++) {
                flag = testMatrix1.getSynapse(r, c).getValue() == 0;
                assertTrue(flag);
            }
        }
    }

    @Test
    public void testTanh() throws Exception {
        Neuron neuron = new Neuron();
        Matrix combine = new Matrix(new Synapse[][]{{new Synapse(0.4)}});
        neuron.setCombineData(combine);
        Process.tanh(neuron);
        double epsilon = 0.0001;
        assertTrue(
                Math.abs(neuron.getActivateData().getSynapse(0, 0).getValue() - 0.379949) < epsilon
                        && neuron.getActivateData().getDimRow() == 1
                        && neuron.getActivateData().getDimCol() == 1);

    }

    @Test
    public void testSigmoid() throws Exception {
        Neuron neuron = new Neuron();
        Matrix combine = new Matrix(new Synapse[][]{{new Synapse(0.3)}});
        neuron.setCombineData(combine);
        Process.sigmoid(neuron);
        double epsilon = 0.0001;
        assertTrue(
                Math.abs(neuron.getActivateData().getSynapse(0, 0).getValue() - 0.57444) < epsilon
                        && neuron.getActivateData().getDimRow() == 1
                        && neuron.getActivateData().getDimCol() == 1);
    }

    @Test
    public void testListToNormalizedInput() {
        Matrix[] testInput = Process.listToNormalizedInput(2, 2, 1);
        double epsilon = 0.001;

        assertTrue(
                Math.abs(testInput[0].getSynapse(0, 0).getValue() - 2.0/3) < epsilon
                        && testInput[0].getDimRow() == 1
                        && testInput[0].getDimCol() == 1);
        assertTrue(
                Math.abs(testInput[1].getSynapse(0, 0).getValue() - 2.0/3) < epsilon
                        && testInput[1].getDimRow() == 1
                        && testInput[1].getDimCol() == 1);
        assertTrue(
                Math.abs(testInput[2].getSynapse(0, 0).getValue() - 1.0/3) < epsilon
                        && testInput[2].getDimRow() == 1
                        && testInput[2].getDimCol() == 1);
        Matrix[] testZeroVector = Process.listToNormalizedInput(0, 0);
        assertTrue(testZeroVector[0].getSynapse(0, 0).getValue() == 0
                && testInput[0].getDimRow() == 1
                && testInput[0].getDimCol() == 1);
        assertTrue(testZeroVector[1].getSynapse(0, 0).getValue() == 0
                && testInput[1].getDimRow() == 1
                && testInput[1].getDimCol() == 1);
    }

    @Test
    public void testListToOutput() {
        Matrix[] testInput = Process.listToOutput(1.5, 7.5, 2.5);


        assertTrue(
                testInput[0].getSynapse(0, 0).getValue() == 1.5
                        && testInput[0].getDimRow() == 1
                        && testInput[0].getDimCol() == 1);
        assertTrue(
                testInput[1].getSynapse(0, 0).getValue() == 7.5
                        && testInput[1].getDimRow() == 1
                        && testInput[1].getDimCol() == 1);
        assertTrue(
                testInput[2].getSynapse(0, 0).getValue() == 2.5
                        && testInput[2].getDimRow() == 1
                        && testInput[2].getDimCol() == 1);
    }

    @Test
    public void testAutoDifferentiate() {
        // testing a more complicated two-variable function involving natural logarithm:
        // f(x,y) = (e^x)/(lny)
        // which is equivalent to f(x,y) = (e^x) * (lny)^-1
        // differentiating with respect to x gives (lny)^-1 * e^x, which is about 207.321 for (x,y) = (6,7)
        // differentiating with respect to y gives (e^x)*(-1)*((lny)^-2)*y^-1 which is about -15.220 for (x,y) = (6,7)
        Synapse x = new Synapse(6);
        Synapse y = new Synapse(7);
        Synapse f1 = Synapse.multiply(
                Synapse.exp(x),
                Synapse.pow(Synapse.ln(y), -1)
        );
        Synapse f2 = Synapse.minus(Synapse.multiply(new Synapse(2), x), Synapse.multiply(new Synapse(5), y));
        Synapse[][] synapses = new Synapse[1][2];
        synapses[0][0] = f1;
        synapses[0][1] = f2;
        Process.autoDifferentiate(new Matrix(synapses));
        double epsilon = 0.001;
        assertTrue(Math.abs(x.getDerivative() - (207.321 + 2)) < epsilon);
        assertTrue(Math.abs(y.getDerivative() - (-15.220 - 5)) < epsilon);
    }


}
