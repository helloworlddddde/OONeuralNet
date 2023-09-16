package ui;


import model.neuralnetwork.Layer;
import model.neuralnetwork.Neuron;
import model.operation.Process;
import model.neuralnetwork.Network;
import model.tensor.Matrix;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.category.CategoryDataset;
import org.jfree.data.category.DefaultCategoryDataset;
import persistence.JsonReader;
import persistence.JsonWriter;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.WindowEvent;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;



// Main class for running the application with GUI
public class Main {


    private static JFrame frame = new JFrame();
    private static Network network;
    private static JScrollPane networkPane;
    private static JPanel panel = new JPanel();
    private static JTextField inputText = new JTextField("Input");
    private static JMenuBar mb;
    private static JButton inputButton = new JButton(new AbstractAction("Input Data") {
        // REQUIRES: data in inputText must have same number of elements as the number of neurons in layer 0
        // of the network
        // MODIFIES: nothing
        // EFFECTS: input vector data and plot the output vector on button click
        public void actionPerformed(ActionEvent e) {
            double[] tempData = new double[network.getLayer(0).getSize()];
            String[] stringData = inputText.getText().split(",", 0);
            for (int j = 0; j < stringData.length; j++) {
                tempData[j] = Double.parseDouble(stringData[j]);
            }
            //System.out.println("Output: ");
            try {
                //System.out.println(network.output(Process.listToNormalizedInput(tempData)));
                Network.logInput(inputText.getText());
                String[] outputLines = network.output(Process.listToNormalizedInput(tempData)).replace("[", "")
                        .replace("]", "").split("\\r?\\n");
                DefaultCategoryDataset dataset = new DefaultCategoryDataset();
                JFreeChart barChart = ChartFactory.createBarChart("Output", "Output",
                        "Value", createDataset(outputLines), PlotOrientation.VERTICAL,
                        true, true, false);
                ChartPanel chartPanel = new ChartPanel(barChart);
                chartPanel.setPreferredSize(new java.awt.Dimension(560, 367));
                JFrame chartFrame = new JFrame();
                chartFrame.setContentPane(chartPanel);
                chartFrame.setSize(500, 500);
                chartFrame.setVisible(true);
            } catch (Exception exception) {
                exception.printStackTrace();
            }
        }
    });

    private static JTextField loadText = new JTextField("Load File Name");
    private static JButton loadButton = new JButton(new AbstractAction("Load Network") {
        // REQUIRES: nothing
        // MODIFIES: network
        // EFFECTS: load an existing saved network upon button click with name specified in loadText
        public void actionPerformed(ActionEvent e) {
            try {
                loadNetwork(loadText.getText());
                panel.remove(networkPane);
                frame.remove(panel);
                initializeTable();
                frame.repaint();
                frame.revalidate();
            } catch (FileNotFoundException fileNotFoundException) {
                fileNotFoundException.printStackTrace();
            }
        }
    });

    private static JButton resetButton = new JButton(new AbstractAction("Reset Parameters") {
        // REQUIRES: nothing
        // MODIFIES: network
        // EFFECTS: reset the parameters (weights and biases) of the network on button click
        public void actionPerformed(ActionEvent e) {
            panel.remove(networkPane);
            frame.remove(panel);
            network.fullConnect();
            initializeTable();
            frame.repaint();
            frame.revalidate();
        }
    });

    private static JTextField saveText = new JTextField("Save File Name");
    private static JButton saveButton = new JButton(new AbstractAction("Save Network") {
        // REQUIRES: nothing
        // MODIFIES: nothing
        // EFFECTS: save the network with name specified in saveText
        public void actionPerformed(ActionEvent e) {
            try {
                saveNetwork(saveText.getText());
            } catch (FileNotFoundException fileNotFoundException) {
                fileNotFoundException.printStackTrace();
            }
        }
    });

    private static JTextField expectedText = new JTextField("Expected Output");
    private static JButton trainButton = new JButton(new AbstractAction("Train Network") {
        // REQUIRES: nothing
        // MODIFIES: nothing
        // EFFECTS: train the neural network with the vector data specified in inputText and expectedText and
        // add the data to the training table
        public void actionPerformed(ActionEvent e) {
            double[] tempData = new double[network.getLayer(0).getSize()];
            String[] stringData = inputText.getText().split(",", 0);
            for (int j = 0; j < stringData.length; j++) {
                tempData[j] = Double.parseDouble(stringData[j]);
            }
            Matrix[] input = Process.listToNormalizedInput(tempData);
            tempData = new double[network.getLayer(network.getLayers().size() - 1).getSize()];
            stringData = expectedText.getText().split(",", 0);
            for (int j = 0; j < stringData.length; j++) {
                tempData[j] = Double.parseDouble(stringData[j]);
            }

            try {
                network.backProp(Process::crossEntropy, input, Process.listToOutput(tempData), 0.05);
            } catch (Exception exception) {
                exception.printStackTrace();
            }
            panel.remove(networkPane);

            frame.remove(panel);
            initializeTable();
            frame.repaint();
            frame.revalidate();
        }
    });



    // REQUIRES: nothing
    // MODIFIES: nothing
    // EFFECTS: launch the opening menu screen with an "Open Program" button and an image of a multilayer perceptron
    private static void openMenu() throws IOException {
        JFrame openFrame = new JFrame();
        JButton openButton = new JButton(new AbstractAction("Open Program") {
            // REQUIRES: nothing
            // MODIFIES: network
            // EFFECTS: open the neural network GUI on click
            public void actionPerformed(ActionEvent e) {
                openFrame.dispose();
                network = Network.multilayerPerceptron(4, 5, 3, 4, 5, 7);
                initializeGUI();
                initializeTable();

            }
        });
        JPanel openPanel = new JPanel();
        openPanel.add(openButton);
        openFrame.setSize(500, 500);
        BufferedImage myPicture = ImageIO.read(new File("./data/openingImage.png"));
        JLabel picLabel = new JLabel(new ImageIcon(myPicture));
        openPanel.add(picLabel);
        openFrame.add(openPanel);
        openFrame.setVisible(true);
    }


    public static void main(String[] args) throws Exception {
        //new NetworkApp();
        frame.addWindowListener(new java.awt.event.WindowAdapter() {
            // REQUIRES: nothing
            // MODIFIES: nothing
            // EFFECTS: End the neural network when the exit (x) button is pressed
            public void windowClosing(WindowEvent winEvt) {
                Network.endNetwork();
            }
        });
        openMenu();
    }

    // REQUIRES: nothing
    // MODIFIES: nothing
    // EFFECTS: add the given input data to a data set, and return the data set
    private static CategoryDataset createDataset(String[] outputLines) {
        DefaultCategoryDataset dataset = new DefaultCategoryDataset();
        for (int i = 0; i < outputLines.length; i++) {
            Number value = Double.parseDouble(outputLines[i]);
            dataset.addValue(value, "Outputs for input: " + inputText.getText(), i);
        }

        return dataset;
    }

    // REQUIRES: nothing
    // MODIFIES: frame, mb
    // EFFECTS: initialize the menu of the GUI
    private static void initializeGUI() {
        frame.setSize(500, 500);
        frame.setVisible(true);
        frame.add(panel);
        mb = new JMenuBar();
        JMenu x = new JMenu("Menu");
        JMenuItem addLayer = new JMenuItem(new AbstractAction("Add Layer") {
            public void actionPerformed(ActionEvent e) {
                panel.remove(networkPane);
                frame.remove(panel);
                Layer layer = new Layer(4, Process::rowAppend, Process::linear, Process::tanh);
                network.addLayer(layer, network.getLayers().size() - 1);
                network.fullConnect();
                initializeTable();
                frame.repaint();
                frame.revalidate();
            }
        });
        x.add(addLayer);
        mb.add(x);
        addMenuElements();
        frame.setJMenuBar(mb);
    }

    // REQUIRES: nothing
    // MODIFIES: mb
    // EFFECTS: add text and button components to the menu bar
    private static void addMenuElements() {
        mb.add(trainButton);
        mb.add(expectedText);
        mb.add(inputButton);
        mb.add(inputText);
        mb.add(loadButton);
        mb.add(loadText);
        mb.add(saveButton);
        mb.add(saveText);
        mb.add(resetButton);
    }

    // REQUIRES: fileName != null
    // MODIFIES: this
    // EFFECTS: load network from ./data/fileName.json
    private static void loadNetwork(String fileName) throws FileNotFoundException {
        JsonReader jsonReader = new JsonReader("./data/" + fileName + ".json");
        try {
            Network net = jsonReader.read();
            network = net;
        } catch (IOException e) {
            throw new FileNotFoundException();
        }

    }

    // REQUIRES: fileName != null
    // MODIFIES: this
    // EFFECTS: save network to ./data/fileName.json
    private static void saveNetwork(String fileName) throws FileNotFoundException {
        JsonWriter jsonWriter = new JsonWriter("./data/" + fileName + ".json");
        jsonWriter.open();
        jsonWriter.write(network);
        jsonWriter.close();
    }

    // REQUIRES: nothing
    // MODIFIES: networkPane;
    // EFFECTS: generate the table representing the parameters of the network
    private static void initializeTable() {
        List<String[]> dataRows = new ArrayList<String[]>();
        for (int i = 1; i < network.getLayers().size(); i++) {
            for (int j = 0; j < network.getLayer(i).getSize(); j++) {
                String[] data = new String[13];
                data[0] = "L" + i + ":" + "N" + j;
                for (int c = 0; c < network.getLayer(i).getNeuron(j).getWeights().getDimCol(); c++) {
                    data[c + 1] = network.getLayer(i).getNeuron(j).getWeights().getSynapse(0, c).toString();
                }
                data[12] = network.getLayer(i).getNeuron(j).getBiases().getSynapse(0,0).toString();
                dataRows.add(data);
            }
        }
        String[][] tableData = new String[dataRows.size()][];
        for (int r = 0; r < dataRows.size(); r++) {
            tableData[r] = dataRows.get(r);
        }
        String[] columnNames = {"Layer:Neuron","W1", "W2", "W3","W4", "W5", "W6", "W7", "W8", "W9", "W10", "W11", "B"};
        JTable j = new JTable(tableData, columnNames);
        networkPane = new JScrollPane(j);
        networkPane.setPreferredSize(new Dimension(1000, 500));
        panel.add(networkPane);
        frame.add(panel);



    }


}






