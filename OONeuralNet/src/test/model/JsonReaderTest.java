package model;

import model.neuralnetwork.Network;
import model.neuralnetwork.Neuron;
import org.junit.jupiter.api.Test;
import persistence.JsonReader;
import persistence.JsonWriter;

import java.io.IOException;

import static org.junit.jupiter.api.Assertions.*;

public class JsonReaderTest {

    @Test
    void testNonExistentFile() {
        JsonReader reader = new JsonReader("./data/noSuchFile.json");
        try {
            Network network = reader.read();
            fail("IOException expected");
        } catch (IOException e) {
            // pass
        }
    }

    @Test
    void testRead() throws IOException {


        JsonReader jsonReader = new JsonReader("./data/testReadMLP.json");
        try {
            Network loadedNetwork = jsonReader.read();
            assertTrue(loadedNetwork.getLayers().size() == 3);
            assertEquals(0.07533452946344998, loadedNetwork.getLayer(1).getNeuron(0).getBiases().
                    getSynapse(0, 0).getValue(), 0.001);
        } catch (IOException e) {
            fail("IOException not expected");
        }




    }




}
