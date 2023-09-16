package model;

import model.neuralnetwork.Synapse;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class SynapseTest {


    @Test
    public void testPlus() {
        Synapse testSynapse1 = new Synapse(2);
        Synapse testSynapse2 = new Synapse(3);
        Synapse sum = Synapse.plus(testSynapse1, testSynapse2);
        assertTrue(sum.getValue() == 5);

        testSynapse1 = new Synapse(2, false);
        testSynapse2 = new Synapse(3, false);
        sum = Synapse.plus(testSynapse1, testSynapse2);
        assertTrue(sum.getValue() == 5);
    }

    @Test
    public void testMinus() {
        Synapse testSynapse1 = new Synapse(100);
        Synapse testSynapse2 = new Synapse(103, false);
        Synapse difference = Synapse.minus(testSynapse1, testSynapse2);
        assertTrue(difference.getValue() == -3);

        testSynapse1 = new Synapse(100, false);
        testSynapse2 = new Synapse(103, true);
        difference = Synapse.minus(testSynapse1, testSynapse2);
        assertTrue(difference.getValue() == -3);
    }

    @Test
    public void testMultiply(){
        Synapse testSynapse1 = new Synapse(101, true);
        Synapse testSynapse2 = new Synapse(303, false);
        Synapse sum = Synapse.multiply(testSynapse1, testSynapse2);
        assertTrue(sum.getValue() == 101*303);

        testSynapse1 = new Synapse(101, false);
        testSynapse2 = new Synapse(303, true);
        sum = Synapse.multiply(testSynapse1, testSynapse2);
        assertTrue(sum.getValue() == 101*303);

        testSynapse1 = new Synapse(101, true);
        testSynapse2 = new Synapse(303, true);
        sum = Synapse.multiply(testSynapse1, testSynapse2);
        assertTrue(sum.getValue() == 101*303);

        testSynapse1 = new Synapse(101, false);
        testSynapse2 = new Synapse(303, false);
        sum = Synapse.multiply(testSynapse1, testSynapse2);
        assertTrue(sum.getValue() == 101*303);
    }

    @Test
    public void testExp(){
        Synapse testSynapse = new Synapse(3.14159);
        testSynapse = Synapse.exp(testSynapse);
        double epsilon = 0.000000000001;
        assertTrue(Math.abs(testSynapse.getValue() - Math.exp(3.14159)) < epsilon);
        testSynapse = new Synapse(2, false);
        testSynapse = Synapse.exp(testSynapse);
        assertTrue(Math.abs(testSynapse.getValue() - Math.exp(2)) < epsilon);
    }

    @Test
    public void testLn(){
        Synapse testSynapse = new Synapse(2.718282);
        testSynapse = Synapse.ln(testSynapse);
        double epsilon = 0.000000000001;
        assertTrue(Math.abs(testSynapse.getValue() - Math.log(2.718282)) < epsilon);
        testSynapse = new Synapse(3, false);
        testSynapse = Synapse.ln(testSynapse);
        assertTrue(Math.abs(testSynapse.getValue() - Math.log(3)) < epsilon);
    }

    @Test
    public void testPow(){
        Synapse testSynapse = new Synapse(-54);
        testSynapse = Synapse.pow(testSynapse, 3);
        assertTrue(testSynapse.getValue() == -54 * -54 * -54);

        testSynapse = new Synapse(-40, false);
        testSynapse = Synapse.pow(testSynapse, 2);
        assertTrue(testSynapse.getValue() == -40 * -40);
    }

    @Test
    public void testToString(){
        Synapse testSynapse = new Synapse(8);
        assertTrue(testSynapse.toString().equals("8.0"));
    }

    @Test
    public void testAutoDifferentiate(){

        // testing a simple one-variable linear function: product = constant * variable
        // obviously, differentiating with product with respect to variable will give the constant
        Synapse variable = new Synapse(4, true); // computation needed to find derivative
        Synapse constant = new Synapse(3, false); // no computation needed to find derivative
        Synapse product = Synapse.multiply(variable, constant);
        product.autoDifferentiate();
        assertTrue(variable.getDerivative() == 3);
        assertTrue(constant.getDerivative() == 0);

        // testing a simple two-variable linear function: product = variable1 * variable2
        // obviously, differentiating with product with respect to a variable will give the other variable
        Synapse variable1 = new Synapse(10, true); // computation needed to find derivative
        Synapse variable2 = new Synapse(21); // computation needed to find derivative
        product = Synapse.multiply(variable1, variable2);
        product.autoDifferentiate();
        assertTrue(variable1.getDerivative() == 21);
        assertTrue(variable2.getDerivative() == 10);

        // testing a slightly more complicated two-variable linear function:
        // y = c1*x1 + c2*x2 + c3*x1 + c4*x2
        // clearly, with respect to x1, the result will be c1 + c3, and with respect to x2, the result will be c2 + c4
        Synapse c1 = new Synapse(1, false);
        Synapse c2 = new Synapse(2, false);
        Synapse c3 = new Synapse(3.5, false);
        Synapse c4 = new Synapse(5.2, false);
        Synapse x1 = new Synapse(7.7);
        Synapse x2 = new Synapse(8.8);
        Synapse p1 = Synapse.multiply(c1, x1);
        Synapse p2 = Synapse.multiply(c2, x2);
        Synapse p3 = Synapse.multiply(c3, x1);
        Synapse p4 = Synapse.multiply(c4, x2);
        Synapse y = Synapse.plus(Synapse.plus(p1, p2), Synapse.minus(p3, p4));
        y.autoDifferentiate();
        assertTrue(x1.getDerivative() == 1 + 3.5);
        assertTrue(x2.getDerivative() == 2 - 5.2);

        // testing a more complicated one-variable function involving chain rule and product rule:
        // z = x^2 * e^(2x)
        // differentiating symbolically gives 2x * e^2x + x^2 * 2 * e^2x
        // z' is about 655.1778 for x = 2
        Synapse x = new Synapse(2);
        Synapse z = Synapse.multiply(
                Synapse.pow(x, 2),
                Synapse.exp(Synapse.multiply(new Synapse(2), x)));
        z.autoDifferentiate();
        double epsilon = 0.0001;
        assertTrue(Math.abs(x.getDerivative() - 655.1778) < epsilon);

        // testing a more complicated two-variable function involving natural logarithm:
        // f(x,y) = (e^x)/(lny)
        // which is equivalent to f(x,y) = (e^x) * (lny)^-1
        // differentiating with respect to x gives (lny)^-1 * e^x, which is about 207.321 for (x,y) = (6,7)
        // differentiating with respect to y gives (e^x)*(-1)*((lny)^-2)*y^-1 which is about -15.220 for (x,y) = (6,7)
        x = new Synapse(6);
        y = new Synapse(7);
        Synapse f = Synapse.multiply(
                Synapse.exp(x),
                Synapse.pow(Synapse.ln(y), -1)
        );
        f.autoDifferentiate();
        epsilon = 0.001;
        assertTrue(Math.abs(x.getDerivative() - 207.321) < epsilon);
        assertTrue(Math.abs(y.getDerivative() - -15.220) < epsilon);

    }

    @Test
    public void testGradientDescent(){

        Synapse x = new Synapse(6);
        Synapse y = new Synapse(7);
        Synapse f = Synapse.multiply(
                Synapse.exp(x),
                Synapse.pow(Synapse.ln(y), -1)
        );

        f.autoDifferentiate();

        double gradX = x.getDerivative();
        x.gradientDescent(1);
        assertTrue(x.getValue() == 6 - gradX);

        double gradY = y.getDerivative();
        y.gradientDescent(0.5);
        assertTrue(y.getValue() == 7 - 0.5 * gradY);

        f = Synapse.multiply(
                Synapse.exp(x),
                Synapse.pow(Synapse.ln(y), -1)
        );
        double tempValue = x.getValue();
        Synapse z = Synapse.multiply(new Synapse(2), x);
        z.autoDifferentiate();
        x.gradientDescent(1);
        assertTrue(x.getValue() == tempValue - 2);



    }

}
