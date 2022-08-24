package ANN.Activations;

import ANN.Layers.ActivationLayer;

public class Tanh extends ActivationLayer {
    public Tanh() {
        super(
                (a) -> Math.tanh(a),
                (a) -> 1 - (Math.pow(Math.tanh(a), 2))
                //(a) -> a * (1-a)
        );
    }
}
