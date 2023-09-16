package model.neuralnetwork;

import java.util.function.Consumer;


// A Synapse represent the data in a neural network. Each Synapse holds a double value and performs mathematical
// operations with other synapses (plus, minus, multiply) or with itself (exponential, logarithm, power).
// Gradient descent is achieved through automatic differentiation.
// The chain of differentiation is maintained via the Consumer for differentiation function: derivativeFunction.
public class Synapse {

    //<editor-fold desc="Fields of Synapse">
    private static final Consumer<Double> emptyFunction = x -> { };

    private double value;

    private double derivative = 0;

    private Consumer<Double> derivativeFunction;

    private boolean toDerive;
    //</editor-fold>

    //<editor-fold desc="Synapse Constructors">
    // REQUIRES: nothing
    // MODIFIES: this
    // EFFECTS: initialize a variable (subject to data change) Synapse with the specified value
    public Synapse(double value) {
        this.value = value;
        this.toDerive = true;
        this.derivativeFunction = (prevDiff) -> derivative += prevDiff;
    }

    // REQUIRES: nothing
    // MODIFIES: this
    // EFFECTS: initialize either a variable or constant Synapse with the specified value
    public Synapse(double value, Boolean toDerive) {
        this.value = value;
        this.toDerive = toDerive;
        this.derivativeFunction = (prevDerivative) -> derivative += prevDerivative;
    }

    // REQUIRES: derivativeFunction != null
    // MODIFIES: this
    // EFFECTS: initialize a variable or constant Synapse with custom derivative function
    public Synapse(double value, boolean toDerive, Consumer<Double> derivativeFunction) {
        this.value = value;
        this.toDerive = toDerive;
        this.derivativeFunction = derivativeFunction;
    }
    //</editor-fold>

    //<editor-fold desc="Basic accessors and mutators for Synapse">
    public double getDerivative() {
        return this.derivative;
    }

    public double getValue() {
        return this.value;
    }
    //</editor-fold>

    //<editor-fold desc="Text manipulations for Synapse">
    // REQUIRES: nothing
    // MODIFIES: this
    // EFFECTS: returns the value of the Synapse in String form
    public String toString() {
        return Double.toString(value);
    }
    //</editor-fold>

    //<editor-fold desc="Basic Mathematical Operations">
    // REQUIRES: both synapse1 and synapse2 are not null
    // MODIFIES: derivativeFunction of both synapses
    // EFFECTS: return a new synapse with value that is the product
    // of the input synapses' values. Differentiate the product with respect
    // to synapse1 and synapse2, and update their derivativeFunctions correspondingly
    // [∂(yx)/∂x = y, ∂(yx)/∂y = x].
    // Then, pass the reference of synapse1 and synapse2's derivativeFunction to the
    // derivativeFunction of the product.
    public static Synapse multiply(Synapse synapse1, Synapse synapse2) {

        double data = synapse1.value * synapse2.value;

        boolean derive = synapse1.toDerive || synapse2.toDerive;

        Consumer<Double> synapse1Fn = synapse1.toDerive
                ? prevDerivative -> synapse1.derivativeFunction
                        .accept(prevDerivative * synapse2.value)
                : emptyFunction;

        Consumer<Double> synapse2Fn = synapse2.toDerive
                ? prevDerivative -> synapse2.derivativeFunction
                        .accept(prevDerivative * synapse1.value)
                : emptyFunction;

        Consumer<Double> derivativeFunction = prevDerivative -> {
            synapse1Fn.accept(prevDerivative);
            synapse2Fn.accept(prevDerivative);
        };

        return new Synapse(data, derive, derivativeFunction);
    }

    // REQUIRES: synapse1 != null && synapse2 != null
    // MODIFIES: derivativeFunction of both synapses
    // EFFECTS: return a new synapse with value that is the sum
    // of the input synapses' values. Differentiate the sum with respect
    // to synapse1 and synapse2, and update their derivativeFunctions correspondingly.
    // Then, pass the reference of synapse1 and synapse2's derivativeFunction to the
    // derivativeFunction of the sum.
    public static Synapse plus(Synapse synapse1, Synapse synapse2) {

        boolean requiresDerivative = synapse1.toDerive || synapse2.toDerive;

        double data = synapse1.value + synapse2.value;

        Consumer<Double> synapse1Fn = synapse1.toDerive
                ? prevDerivative -> synapse1.derivativeFunction
                        .accept(prevDerivative)
                : emptyFunction;

        Consumer<Double> synapse2Fn = synapse2.toDerive
                ? prevDerivative -> synapse2.derivativeFunction
                        .accept(prevDerivative)
                : emptyFunction;

        Consumer<Double> derivativeFunction = prevDerivative -> {
            synapse1Fn.accept(prevDerivative);
            synapse2Fn.accept(prevDerivative);
        };

        return new Synapse(data, requiresDerivative, derivativeFunction);
    }

    // REQUIRES: synapse1 != null && synapse2 != null
    // MODIFIES: derivativeFunction of both synapses
    // EFFECTS: return a new synapse with value that is the difference
    // of the input synapses' values (value of synapse1 - value of synapse 2).
    // Differentiate the difference with respect to synapse1 and synapse2, and update
    // their derivativeFunctions correspondingly.
    // Then, pass the reference of synapse1 and synapse2's derivativeFunction to the
    // derivativeFunction of the difference.
    public static Synapse minus(Synapse synapse1, Synapse synapse2) {
        return plus(synapse1, multiply(new Synapse(-1), synapse2));
    }

    // REQUIRES: synapse != null
    // MODIFIES: derivativeFunction of synapse
    // EFFECTS: return a new synapse with value that is the value of synapse to the power
    // of powerOf. Differentiate the result with respect to to synapse, and update
    // its derivativeFunctions correspondingly [d(x^n)/dx = nx^(n-1)].
    // Then, pass the reference of synapse's derivativeFunction to the derivativeFunction of the result.
    public static Synapse pow(Synapse synapse, int powerOf) {

        double data = Math.pow(synapse.value, powerOf);

        boolean requiresDerivative = synapse.toDerive;

        Consumer<Double> derivativeFunction = synapse.toDerive
                ? prevDerivative -> synapse.derivativeFunction
                .accept(prevDerivative * powerOf * Math.pow(synapse.value, powerOf - 1))
                : emptyFunction;

        return new Synapse(data, requiresDerivative, derivativeFunction);
    }

    // REQUIRES: synapse != null
    // MODIFIES: derivativeFunction of synapse
    // EFFECTS: return a new synapse with value that is the value of the exponentiation of synapse's value
    // Differentiate the result with respect to to synapse, and update
    // its derivativeFunctions correspondingly [d(e^x)/dx = e^x]. Then, pass the reference of synapse's
    // derivativeFunction to the derivativeFunction of the result.
    public static Synapse exp(Synapse synapse) {
        double data = Math.exp(synapse.value);

        boolean requiresDerivative = synapse.toDerive;

        Consumer<Double> derivativeFunction = synapse.toDerive
                ? prevDerivative -> synapse.derivativeFunction
                .accept(prevDerivative * Math.exp(synapse.value))
                : emptyFunction;

        return new Synapse(data, requiresDerivative, derivativeFunction);
    }

    // REQUIRES: synapse != null
    // MODIFIES: derivativeFunction of synapse
    // EFFECTS: return a new synapse with value that is the value of the natural logarithm of synapse's value
    // Differentiate the result with respect to to synapse, and update
    // its derivativeFunctions correspondingly [d(lnx)/dx = 1/x]. Then, pass the reference of synapse's
    // derivativeFunction to the derivativeFunction of the result.
    public static Synapse ln(Synapse synapse) {

        double data = Math.log(synapse.value);

        boolean requiresDerivative = synapse.toDerive;

        Consumer<Double> derivativeFunction = synapse.toDerive
                ? prevDerivative -> synapse.derivativeFunction
                .accept(prevDerivative * 1 / synapse.value)
                : emptyFunction;

        return new Synapse(data, requiresDerivative, derivativeFunction);
    }
    //</editor-fold>

    //<editor-fold desc="Differentiation Operations">
    // REQUIRES: nothing
    // MODIFIES: derivative of synapses with derivativeFunction previously linked to the synapse calling
    // this method
    // EFFECTS: differentiate the synapse calling this method with respect to all other variable synapse
    // with derivativeFunction linked to this synapse's derivativeFunction. The variable synapses will have its
    // derivative value updated corresponding to its partial differentiation.
    public void autoDifferentiate() {
        derivativeFunction.accept(1.0);
    }

    // REQUIRES: nothing
    // MODIFIES: the value of the synapse calling this method
    // EFFECTS: subtract learningRate * derivative to the value of the synapse calling this method
    public void gradientDescent(double learningRate) {
        value -= learningRate * derivative;
        this.derivative = 0;
        this.derivativeFunction = (prevDerivative) -> derivative += prevDerivative;
    }
    //</editor-fold>


}
