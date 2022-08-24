package ANN.Loss;

public class MeanSquaredError {

    public static double mse(double[] expected, double[] predicted) {
        double mse = 0;

        for (int i = 0; i < expected.length; i++) {
            double err =  expected[i] - predicted[i];
            mse += err * err;
        }

        return mse/expected.length;
    }

    public static double[] mse_prime(double[] expected, double[] predicted) {
        double[] gradient = new double[expected.length];

        for (int i = 0; i < expected.length; i++) {
            gradient[i] = 2 * (predicted[i] - expected[i]) / expected.length;
        }

        return gradient;
    }

}
