package persistence;

import model.neuralnetwork.Layer;
import model.neuralnetwork.Network;
import model.neuralnetwork.Neuron;
import model.tensor.Matrix;
import org.json.JSONArray;
import org.json.JSONObject;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.stream.Stream;

// Specifications and general structure are based on https://github.students.cs.ubc.ca/CPSC210/JsonSerializationDemo
// Represents a reader that reads workroom from JSON data stored in file
public class JsonReader {
    private String source;

    // Specifications and general structure are based on https://github.students.cs.ubc.ca/CPSC210/JsonSerializationDemo
    // EFFECTS: constructs reader to read from source file
    public JsonReader(String source) {
        this.source = source;
    }

    // Specifications and general structure are based on https://github.students.cs.ubc.ca/CPSC210/JsonSerializationDemo
    // EFFECTS: reads Network from file and returns it;
    // throws IOException if an error occurs reading data from file
    public Network read() throws IOException {
        String jsonData = readFile(source);
        JSONObject jsonObject = new JSONObject(jsonData);
        return parseNetwork(jsonObject);
    }

    // Specifications and general structure are based on https://github.students.cs.ubc.ca/CPSC210/JsonSerializationDemo
    // EFFECTS: reads source file as string and returns it
    private String readFile(String source) throws IOException {
        StringBuilder contentBuilder = new StringBuilder();

        try (Stream<String> stream = Files.lines(Paths.get(source), StandardCharsets.UTF_8)) {
            stream.forEach(s -> contentBuilder.append(s));
        }

        return contentBuilder.toString();
    }

    // Specifications and general structure are based on https://github.students.cs.ubc.ca/CPSC210/JsonSerializationDemo
    // EFFECTS: parses Network from JSON object and returns it
    private Network parseNetwork(JSONObject jsonObject) {
        JSONArray layers = jsonObject.getJSONArray("Layers");
        int[] layerSizes = new int[layers.length()];
        for (int i = 0; i < layers.length(); i++) {
            layerSizes[i] = (int) layers.getJSONObject(i).get("size");
        }
        Network network = Network.multilayerPerceptron(layerSizes);
        for (int i = 1; i < network.getLayers().size(); i++) {
            JSONArray neurons = (JSONArray) layers.getJSONObject(i).get("Neurons");
            for (int j = 0; j < network.getLayer(i).getSize(); j++) {
                Matrix weights = Matrix.stringToMatrix((String) neurons.getJSONObject(j).get("Weights"));
                Matrix biases = Matrix.stringToMatrix((String) neurons.getJSONObject(j).get("Biases"));
                network.getLayer(i).getNeuron(j).setWeights(weights);
                network.getLayer(i).getNeuron(j).setBiases(biases);
            }
        }

        return network;
    }

}