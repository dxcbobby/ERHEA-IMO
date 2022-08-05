package DL;

import java.io.File;

import org.deeplearning4j.nn.conf.layers.*;
import org.nd4j.linalg.activations.Activation;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;

import org.deeplearning4j.util.ModelSerializer;

import org.nd4j.linalg.api.ndarray.INDArray;

import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;

import org.nd4j.linalg.lossfunctions.LossFunctions;

import org.deeplearning4j.nn.conf.GradientNormalization;

import javax.swing.*;

import enumerate.Action;
import util.Parameter;

import java.util.*;

public class DeepModel {
	static private MultiLayerNetwork net;
	static private int seed = 123; // seed is 123;

	static private double learningRate = 0.001;

	static private HashMap<Action, Integer> actToIndex;
	final static private int numOutputs = 56;
	final static private int n_epochs = 3;

	final static private String directory = "./data/aiData/ERHEA00";
	private static String oppName;
	private static String myName;
	private static String oppModelName;

	private static String lossType;

	public DeepModel(int input_size, String myName, String oppName, String lossType) {
		this.myName = myName;
		this.oppName = oppName;
		this.lossType = lossType;
		this.oppModelName = "oppModel.zip";
		this.oppModelName = this.lossType + "-" + this.oppModelName;
		// network learning model
		System.out.println("Create " + this.lossType + " Model!");

		MultiLayerConfiguration conf = getConfig(input_size);
		net = new MultiLayerNetwork(conf);
		net.init();
		net.setListeners(new ScoreIterationListener(1));

		actToIndex = new HashMap<Action, Integer>();
		for (int i = 0; i < Action.values().length; i++) {
			actToIndex.put(Action.values()[i], i);
		}

	}

// num input: my HP, my Energy, my Posx, my Posy, my state, diff posx, opp HP, opp Energy, opp Posx, opp Posy, opp State, 
	// my Last Act,opp Last Act
	private MultiLayerConfiguration getConfig(int input_size) {

		if (lossType.toLowerCase() == "pi") {
//		   return new NeuralNetConfiguration.Builder()
//				   .seed(seed)
////				   .weightInit(WeightInit.XAVIER)
//				   .weightInit(WeightInit.ONES)
//				   .biasInit(0.0)
//				   .gradientNormalization(GradientNormalization.ClipL2PerLayer)
//				   .gradientNormalizationThreshold(1.0)
//				   .updater(new Adam(learningRate, 0.99, 0.999, 0.00000001)) //
//				   .list()
//				   .layer(0,new OutputLayer.Builder(new PILoss())  // best performance: PILoss()
//						   .activation(Activation.SOFTMAX)   // pi: softmax, q: identity
//						   .nIn(input_size).nOut(numOutputs).build())
//				   .build();
			int inputNum = Parameter.inputNum;
			int outputNum = Parameter.outputNum;
			int hiddenUnit = 0;
			Activation activation = Activation.RELU;
			System.out.println("Input: " + inputNum + ", Hidden: " + hiddenUnit + ", Out: " + outputNum);
			return new NeuralNetConfiguration.Builder().seed(seed).list()
					.layer(0, new OutputLayer.Builder().nIn(inputNum).activation(Activation.SOFTMAX).nOut(outputNum)
							.build())
					.build();

		} else if (lossType.toLowerCase() == "sl") {
			return new NeuralNetConfiguration.Builder().seed(seed)
//				   .weightInit(WeightInit.XAVIER)
					.weightInit(WeightInit.ONES).gradientNormalization(GradientNormalization.ClipL2PerLayer)
					.gradientNormalizationThreshold(1.0).updater(new Adam(learningRate, 0.99, 0.999, 0.00000001)) //
					.list().layer(0, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT) // best performance:
																								// PILoss()
							.activation(Activation.SOFTMAX) // pi: softmax, q: identity
							.nIn(input_size).nOut(numOutputs).build())
					.build();

		}
		System.err.printf("Not right loss:%s \n", lossType);
		return null;

	}

	public int forward(float[] input_info) {
		int num_out = 0;
		INDArray input;
		INDArray out;

		input = Nd4j.create(input_info);
		out = net.output(input);

		INDArray argmax;
		// System.out.println("Out:" + out);
		argmax = Nd4j.argMax(out, 1);
		num_out = (int) (argmax.getInt(0));

		return num_out;
	}

	public int forward(float[] input_info, LinkedList<Action> validActs) {

		int num_out = 0;
		double maxV = -99999999.0;
		double val;
		INDArray input;
		INDArray out;

		input = Nd4j.create(input_info);
		out = net.output(input);

//		double[] ds = out.toDoubleVector();
//		for (Action act : validActs) {
//			int i = actToIndex.get(act);
//			val = ds[i];
//			if (val > maxV) {
//				maxV = val;
//				num_out = i;
//			}
//		}
		INDArray argmax;
		// System.out.println("Out:" + out);
		argmax = Nd4j.argMax(out, 1);
		num_out = (int) (argmax.getInt(0));

		return num_out;

	}

	private String getPathName() {
		String myname;
		String oppname;
		myname = this.myName == "" ? "" : this.myName + "-";
		oppname = this.oppName == "" ? "" : this.oppName + "-";
		return this.directory + "/" + myname + oppname + this.oppModelName;
	}

	public void save() {
		File file = new File(this.directory);
		if (!file.exists()) {
			file.mkdir();
		}

		String pathName;
		pathName = getPathName();
		try {
			System.out.println("Save Model!");
			ModelSerializer.writeModel(net, new File(pathName), true); // false is not saved optimization parameters.
		} catch (Exception e) {
			e.printStackTrace();
		}

	}

	public void load() {
		String pathName = Parameter.modelPathName;
//   	   pathName = getPathName();
		try {
			File loadFile = new File(pathName);
			if (loadFile.exists()) {
				System.out.println("Loaded Model!");
				net = ModelSerializer.restoreMultiLayerNetwork(loadFile);
			} else {
				System.out.println("Could not find out the model!");
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

}
