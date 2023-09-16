package model;

import model.tensor.Matrix;
import model.neuralnetwork.Synapse;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class MatrixTest {

    String str1Input;
    Matrix mat1;
    String str1Output;

    String str2Input;
    Matrix mat2;
    String str2Output;


    @BeforeEach
    public void runBefore() {

        str1Input = "[1, 2, 3], [4, 5, 6]";
        // a 2x3 Matrix
        mat1 = new Matrix(new Synapse[][]{
                {new Synapse(1), new Synapse(2), new Synapse(3)},
                {new Synapse(4), new Synapse(5), new Synapse(6)}
        });
        str1Output = "[ 1.0 2.0 3.0 ]" + "\n" + "[ 4.0 5.0 6.0 ]" + "\n";

        str2Input = "[1, 2, 3, 4, 5]";
        // a 1x5 Matrix
        mat2 = new Matrix(new Synapse[][]{
                {new Synapse(1), new Synapse(2), new Synapse(3), new Synapse(4), new Synapse(5)}
        });
        str2Output = "[ 1.0 2.0 3.0 4.0 5.0 ]" + "\n";

    }

    @Test
    public void testToString() {
        assertTrue(str1Output.equals(mat1.toString()));
        assertTrue(str2Output.equals(mat2.toString()));
        String test = mat1.toString2();
    }

    @Test
    public void testStringToMatrix() {
        Matrix toCompare1 = Matrix.stringToMatrix(str1Input);
        boolean flag;
        for(int r = 0; r < mat1.getDimRow(); r++){
            for(int c = 0; c < mat1.getDimCol(); c++){
                flag = (mat1.getSynapse(r, c).getValue() == toCompare1.getSynapse(r, c).getValue());
                assertTrue(flag);
                // check if values of Synapses are all equal
            }
        }

        assertTrue(
                toCompare1.getDimCol() == mat1.getDimCol() && // check if dimensions are equal
                        toCompare1.getDimRow() == mat1.getDimRow());

        Matrix toCompare2 = Matrix.stringToMatrix(str2Input);
        for(int r = 0; r < mat2.getDimRow(); r++){
            for(int c = 0; c < mat2.getDimCol(); c++){
                flag = (mat2.getSynapse(r, c).getValue() == toCompare2.getSynapse(r, c).getValue());
                assertTrue(flag);
                // check if values of Synapses are all equal
            }
        }

        assertTrue(
                toCompare2.getDimCol() == mat2.getDimCol() && // check if dimensions are equal
                        toCompare2.getDimRow() == mat2.getDimRow());
    }

    @Test
    public void testSetData() {
        Synapse testSynapse = new Synapse(5);
        mat1.setData(1, 2, testSynapse);
        assertTrue(testSynapse.equals(mat1.getSynapse(1, 2)));
        mat2.setData(0, 0, testSynapse);
        mat2.setData(0, 4, testSynapse);
        assertTrue((testSynapse.equals(mat2.getSynapse(0,0))) && testSynapse.equals(mat2.getSynapse(0, 4)));
    }

    @Test
    public void testGetSynapse() {
        Synapse testSynapse1 = new Synapse(0.5);
        Synapse testSynapse2 = new Synapse(-0.5);
        Synapse[][] testSynapses = new Synapse[][]{{testSynapse1}, {testSynapse2}};
        Matrix testMatrix = new Matrix(testSynapses);
        assertTrue(testMatrix.getSynapse(0, 0).equals(testSynapse1));
        assertTrue(testMatrix.getSynapse(1, 0).equals(testSynapse2));

    }
}
