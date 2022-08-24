package ANN.Layers;

public class Dense extends Layer {
    public int inputSize;
    public int outputSize;

    public double[] inputs;
    public double[] outputs;

    public double[][] weights;
    public double[] bias;

    public Dense(int inputSize, int outputSize) {
        this.inputSize = inputSize;
        this.outputSize = outputSize;

        this.weights = new double[outputSize][inputSize];
        this.bias = new double[outputSize];

        for (double[] inputs : weights) {
            for (int weightIdx = 0; weightIdx < inputs.length; weightIdx++) {
                inputs[weightIdx] = Math.random() - 0.5;
            }
        }

        for (int biasIdx = 0; biasIdx < bias.length; biasIdx++) {
            bias[biasIdx] = Math.random() - 0.5;
        }
    }

    @Override
    public double[] FeedForward(double[] input) {
        this.inputs = input;

        double[] out = new double[outputSize];

        for (int outputIdx = 0; outputIdx < outputSize; outputIdx++) {
            double sum = 0;
            for (int inputIdx = 0; inputIdx < inputSize; inputIdx++) {
                //System.out.println("inputs size: " + inputs.length + " weights dimensions: " + weights.length + "x" + weights[0].length);
                sum += inputs[inputIdx] * weights[outputIdx][inputIdx];
            }
            out[outputIdx] = sum + bias[outputIdx];
        }
        this.outputs = out;
        return out;
    }

    @Override
    public double[] BackPropagate(double[] output_gradient, double l_rate) {
        double[][] weights_gradient = new double[outputSize][inputSize];
        for (int outputIdx = 0;  outputIdx < outputSize; outputIdx++) {
            for (int inputIdx = 0; inputIdx < inputSize; inputIdx++) {
                weights_gradient[outputIdx][inputIdx] = output_gradient[outputIdx] * inputs[inputIdx];
                this.weights[outputIdx][inputIdx] -= l_rate * weights_gradient[outputIdx][inputIdx];
            }
        }


        double[] bias_gradient = output_gradient;
        for (int outputIdx = 0;  outputIdx < outputSize; outputIdx++) {
            this.bias[outputIdx] -= l_rate * bias_gradient[outputIdx];
        }

        double[] input_gradient = new double[inputSize];
        for (int inputIdx = 0; inputIdx < inputSize; inputIdx++) {
            double ith = 0;
            for (int outputIdx = 0; outputIdx < outputSize; outputIdx++) {
                ith += output_gradient[outputIdx] * weights_gradient[outputIdx][inputIdx];
            }
            input_gradient[inputIdx] = ith;
        }

        return input_gradient;
    }

    @Override
    public String printInputs() {
        String out = "";
        for (int i = 0; i < inputs.length; i++) {
            out += inputs[i] + " ";
        }
        return out;
    }

    @Override
    public String printOutputs() {
        String out = "";
        for (int i = 0; i < outputs.length; i++) {
            out += outputs[i] + " ";
        }
        return out;
    }

    @Override
    public String printBias() {
        if (bias != null) {
            String out = "";
            for (int i = 0; i < outputs.length; i++) {
                out += outputs[i] + " ";
            }
            return out;
        }
        return null;
    }

    @Override
    public String printWeights() {
        if (bias != null) {
            String out = "";
            for (int i = 0; i < outputs.length; i++) {
                for (int j = 0; j < inputs.length; j++) {
                    out += weights[i][j] + " ";
                }
                out += "\n";
            }
            return out;
        }
        return null;
    }
}
