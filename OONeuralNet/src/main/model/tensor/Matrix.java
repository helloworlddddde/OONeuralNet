package model.tensor;

import model.neuralnetwork.Synapse;

import java.util.ArrayList;

// A Matrix is an m x n tensor. It takes a 2D array of Synapse as its data and deals with basic String manipulation and
// access / modification of those Synapse.
public class Matrix {


    //<editor-fold desc="Fields of Matrix">
    private int dimRow;
    private int dimCol;
    private Synapse[][] synapses;
    //</editor-fold>

    //<editor-fold desc="Matrix constructors">
    // REQUIRES: data is not null
    // MODIFIES: this
    // EFFECTS: create a new m x n Matrix with the given data
    public Matrix(Synapse[][] synapses) {
        this.synapses = synapses;
        dimRow = synapses.length;
        dimCol = synapses[0].length;
    }
    //</editor-fold>

    //<editor-fold desc="Basic accessors and mutators for Matrix">
    public int getDimRow() {
        return this.dimRow;
    }

    public int getDimCol() {
        return this.dimCol;
    }

    // REQUIRES: nothing
    // MODIFIES: this
    // EFFECTS: return the synapse at rth row and cth column in synapses (0-based indexing)
    public Synapse getSynapse(int r, int c) {
        return synapses[r][c];
    }

    // REQUIRES: input synapse is not null
    // MODIFIES: this
    // EFFECTS: replace the synapse at rth row and cth column with a another synapse (0-based indexing)
    public void setData(int r, int c, Synapse synapse) {
        synapses[r][c] = synapse;
    }
    //</editor-fold>

    //<editor-fold desc="Text manipulations for Matrix">
    // REQUIRES: nothing
    // MODIFIES: this
    // EFFECTS: returns the Matrix in text (String) form
    @Override
    public String toString() {
        String output = "";
        for (int r = 0; r < dimRow; r++) {
            output += "[ ";
            for (int c = 0; c < dimCol; c++) {
                output += getSynapse(r, c).toString() + " ";
            }
            output += "]" + "\n";
        }
        return output;
    }

    // REQUIRES: input is not null
    // MODIFIES: this
    // EFFECTS: convert a user typed matrix to a Matrix
    public static Matrix stringToMatrix(String input) {
        ArrayList<Synapse[]> tempArr = new ArrayList<Synapse[]>();
        int index1 = 0;
        int index2;
        for (int i = 0; i < input.length(); i++) {
            if (input.charAt(i) == '[') {
                index1 = i;
            }
            if (input.charAt(i) == ']') {
                index2 = i;
                String[] temp1 = input.substring(index1 + 1, index2).split(",", 0);
                Synapse[] temp2 = new Synapse[temp1.length];
                for (int k = 0; k < temp1.length; k++) {
                    temp2[k] = new Synapse(Double.parseDouble(temp1[k]));
                }
                tempArr.add(temp2);
            }
        }
        Synapse[][] result = new Synapse[tempArr.size()][];
        for (int r = 0; r < tempArr.size(); r++) {
            result[r] = tempArr.get(r);
        }
        return new Matrix(result);
    }

    // REQUIRES: nothing
    // MODIFIES: this
    // EFFECTS: returns the Matrix in text (String) form
    public String toString2() {
        String output = "";
        for (int r = 0; r < dimRow; r++) {
            output += "[ ";
            for (int c = 0; c < dimCol - 1; c++) {
                output += getSynapse(r, c).toString() + ", ";
            }
            output += getSynapse(r, dimCol - 1);
            output += "]";
            if (r < dimRow - 1) {
                output += ",";
            }
        }
        return output;
    }
    //</editor-fold>






}