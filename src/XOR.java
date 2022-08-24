import ANN.Activations.ReLU;
import ANN.Activations.Tanh;
import ANN.Layers.Dense;
import ANN.Layers.Layer;
import ANN.Loss.MeanSquaredError;

public class XOR {
    public static double[][] xTrain = new double[][] {
            {0, 0},
            {0, 1},
            {1, 0},
            {1, 1}
    };

    public static double[][] yTrain = new double[][] {
            {0},
            {1},
            {1},
            {0}
    };

    public static Layer[] network = new Layer[]  {
        new Dense(2, 3),
        new Tanh(),
        new Dense(3, 1),
        new Tanh(),
    };

    public static int epochs = 1000;
    public static double l_rate = 0.1;

    public static void main(String[] args) {


        for (int epoch = 1; epoch <= epochs; epoch++) {
            double error = 0;

            for (int sample = 0; sample < xTrain.length; sample++) {
                double[] output =  xTrain[sample];
                //System.out.println("sample length: " + xTrain[sample].length);

                for (Layer layer : network) {
                    output = layer.FeedForward(output);
                }

                error += MeanSquaredError.mse(yTrain[sample], output);

                double[] gradient = MeanSquaredError.mse_prime(yTrain[sample], output);

                for (int layerIdx = network.length-1; layerIdx >= 0; layerIdx--) {
                    gradient = network[layerIdx].BackPropagate(gradient, l_rate);
                }
            }

            error /= xTrain.length;
            System.out.println("Epoch: " + epoch + "/" + epochs + ", Error: " + error);
        }
        System.out.println("\n\n");

        for (int sample = 0; sample < xTrain.length; sample++)  {
            System.out.println("Network predicting the output of x = {" + xTrain[sample][0] +  ", " +xTrain[sample][1] + "}, expected output of " + yTrain[sample][0]);

            double[] output =  xTrain[sample];
            System.out.println("initial: " + printArr(output));
            for (Layer layer : network) {

                //System.out.println(layer.printWeights());
                //System.out.println(layer.printBias());
            }

            for (Layer layer : network) {
                output = layer.FeedForward(output);
                System.out.println("layer " + layer.getClass().getName() + "   " + printArr(output)) ;
            }

            double[] gradient = MeanSquaredError.mse_prime(yTrain[sample], output);
            System.out.println("error: " + printArr(gradient));
            for (int layerIdx = network.length-1; layerIdx >= 0; layerIdx--) {
                gradient = network[layerIdx].BackPropagate(gradient, l_rate);
                System.out.println("error: " + printArr(gradient));
            }

            System.out.println();
        }
    }

    public static String printArr(double[] arr) {
        String output = "";
        for (int i = 0; i < arr.length; i++) {
            output += arr[i] + " ";
        }

        return output;
    }
}
