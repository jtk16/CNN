package ANN.Layers;

import java.util.function.Function;

public class ActivationLayer extends Layer {
    public Function<Double, Double> activation;
    public Function<Double, Double> dActivation;

    public ActivationLayer(Function<Double, Double> activation, Function<Double, Double> dActivation) {
        this.activation = activation;
        this.dActivation = dActivation;
    }

    public double[] FeedForward(double[] input) {
        this.inputs = input;
        this.outputs = new double[input.length];

        double[] out = new double[outputs.length];
        for (int outputIdx = 0;  outputIdx < outputs.length; outputIdx++) {
            out[outputIdx] = activation.apply(input[outputIdx]);
        }

        return out;
    }

    public double[] BackPropagate(double[] output_gradient, double l_rate) {

        double[] out = new double[inputs.length];
        for (int inputIdx = 0;  inputIdx < outputs.length; inputIdx++) {
            out[inputIdx] = output_gradient[inputIdx] * dActivation.apply(inputs[inputIdx]);
        }

        return out;
    }

}
