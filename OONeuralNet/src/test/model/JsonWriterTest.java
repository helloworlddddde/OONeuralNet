package model;

import model.neuralnetwork.Network;
import model.neuralnetwork.Neuron;
import org.junit.jupiter.api.Test;
import persistence.JsonReader;
import persistence.JsonWriter;

import java.io.FileNotFoundException;
import java.io.IOException;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class JsonWriterTest {


    @Test
    void testWrite() throws IOException {
        JsonWriter jsonWriter = new JsonWriter("./data/testWriteMLP.json");
        jsonWriter.open();
        Network network = Network.multilayerPerceptron(4, 4, 4);
        jsonWriter.write(network);
        jsonWriter.close();

        JsonReader jsonReader = new JsonReader("./data/testWriteMLP.json");
        Network loadedNetwork = jsonReader.read();

        assertTrue(loadedNetwork.getLayers().size() == 3);
        for(int i = 0; i < loadedNetwork.getLayers().size(); i++) {
            assertTrue(loadedNetwork.getLayer(i).getSize() == 4);
        }

        for (int i = 1; i < loadedNetwork.getLayers().size(); i++) {
            for (int j = 0; j < loadedNetwork.getLayer(i).getNeurons().size(); j++) {
                Neuron neuron = network.getLayer(i).getNeuron(j);
                Neuron loadedNeuron = loadedNetwork.getLayer(i).getNeuron(j);
                int row = loadedNeuron.getWeights().getDimRow();
                int col = loadedNeuron.getWeights().getDimCol();
                for (int r = 0; r < row; r++) {
                    for (int c = 0; c < col; c++) {
                        assertTrue(loadedNeuron.getWeights().getSynapse(r, c).getValue()
                                == neuron.getWeights().getSynapse(r, c).getValue());
                    }
                }
                row = loadedNeuron.getBiases().getDimRow();
                col = loadedNeuron.getBiases().getDimCol();
                for (int r = 0; r < row; r++) {
                    for (int c = 0; c < col; c++) {
                        assertTrue(loadedNeuron.getBiases().getSynapse(r, c).getValue()
                                == neuron.getBiases().getSynapse(r, c).getValue());
                    }
                }
            }
        }

    }
}
