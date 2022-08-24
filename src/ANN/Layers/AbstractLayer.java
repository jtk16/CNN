package ANN.Layers;

public interface AbstractLayer {

    double[] FeedForward(double[] input);

    double[] BackPropagate(double[] output_gradient, double l_rate);
}
