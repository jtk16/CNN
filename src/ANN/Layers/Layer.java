package ANN.Layers;

public class Layer implements AbstractLayer {
    double[] inputs = null;
    double[] outputs = null;

    public double[][] weights = null;
    public double[] bias = null;

    public Layer() {

    }

    @Override
    public double[] FeedForward(double[] input) {
        return new double[0];
    }

    @Override
    public double[] BackPropagate(double[] output_gradient, double l_rate) {
        return new double[0];
    }

    public String printInputs() {
        String out = "";
        for (int i = 0; i < inputs.length; i++) {
            out += inputs[i] + " ";
        }
        return out;
    }

    public String printOutputs() {
        String out = "";
        for (int i = 0; i < outputs.length; i++) {
            out += outputs[i] + " ";
        }
        return out;
    }

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
