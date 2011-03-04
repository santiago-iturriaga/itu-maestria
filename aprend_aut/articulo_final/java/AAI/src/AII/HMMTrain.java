package AII;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Reader;

import java.util.ArrayList;
import java.util.Random;
import java.util.logging.Logger;
import java.util.regex.Pattern;

import cc.mallet.types.Alphabet;
import cc.mallet.types.AugmentableFeatureVector;
import cc.mallet.types.FeatureSequence;
import cc.mallet.types.FeatureVector;
import cc.mallet.types.FeatureVectorSequence;
import cc.mallet.types.Instance;
import cc.mallet.types.InstanceList;
import cc.mallet.types.LabelAlphabet;
import cc.mallet.types.LabelSequence;
import cc.mallet.types.Sequence;

import cc.mallet.fst.CRF;
import cc.mallet.fst.CRFTrainerByLabelLikelihood;
import cc.mallet.fst.HMM;
import cc.mallet.fst.HMMTrainerByLikelihood;
import cc.mallet.fst.MaxLatticeDefault;
import cc.mallet.fst.Transducer;
import cc.mallet.fst.TransducerEvaluator;
import cc.mallet.fst.ViterbiWriter;
import cc.mallet.pipe.Pipe;
import cc.mallet.pipe.iterator.LineGroupIterator;

import cc.mallet.util.CommandOption;
import cc.mallet.util.MalletLogger;

public class HMMTrain {
	private static Logger logger = MalletLogger.getLogger(HMMTrain.class
			.getName());

	private HMMTrain() {
	}

	public static class SimpleTaggerSentence2FeatureSequence extends Pipe {
		private static final long serialVersionUID = -2059308802200728624L;

		/**
		 * Creates a new <code>SimpleTaggerSentence2FeatureVectorSequence</code>
		 * instance.
		 */
		public SimpleTaggerSentence2FeatureSequence() {
			super(new Alphabet(), new LabelAlphabet());
		}

		/**
		 * Parses a string representing a sequence of rows of tokens into an
		 * array of arrays of tokens.
		 * 
		 * @param sentence
		 *            a <code>String</code>
		 * @return the corresponding array of arrays of tokens.
		 */
		private String[][] parseSentence(String sentence) {
			String[] lines = sentence.split("\n");
			String[][] tokens = new String[lines.length][];
			for (int i = 0; i < lines.length; i++)
				tokens[i] = lines[i].split(" ");
			return tokens;
		}

		public Instance pipe(Instance carrier) {
			Object inputData = carrier.getData();
			Alphabet features = getDataAlphabet();
			
			LabelAlphabet labels;
			LabelSequence target = null;
			String[][] tokens;

			if (inputData instanceof String)
				tokens = parseSentence((String) inputData);
			else if (inputData instanceof String[][])
				tokens = (String[][]) inputData;
			else
				throw new IllegalArgumentException(
						"Not a String or String[][]; got " + inputData);

			FeatureVector[] fvs = new FeatureVector[tokens.length];
			if (isTargetProcessing()) {
				labels = (LabelAlphabet) getTargetAlphabet();
				target = new LabelSequence(labels, tokens.length);
			}

			for (int l = 0; l < tokens.length; l++) {
				int nFeatures;

				if (isTargetProcessing()) {
					if (tokens[l].length < 1)
						throw new IllegalStateException(
								"Missing label at line " + l + " instance "
										+ carrier.getName());
					nFeatures = tokens[l].length - 1;
					target.add(tokens[l][nFeatures]);
				} else {
					nFeatures = tokens[l].length;
				}

				ArrayList<Integer> featureIndices = new ArrayList<Integer>();
				for (int f = 0; f < nFeatures; f++) {
					int featureIndex = features.lookupIndex(tokens[l][f]);
					if (featureIndex >= 0) {
						featureIndices.add(featureIndex);
					}
				}

				int[] featureIndicesArr = new int[featureIndices.size()];
				for (int index = 0; index < featureIndices.size(); index++) {
					featureIndicesArr[index] = featureIndices.get(index);
				}
		       	fvs[l] = new FeatureVector(features, featureIndicesArr);
			}

			carrier.setData(new FeatureSequence(features, fvs));

			if (isTargetProcessing()) {
				carrier.setTarget(target);
			} else {
				carrier.setTarget(new LabelSequence(getTargetAlphabet()));
			}

			return carrier;
		}
	}

	public static HMM train(InstanceList training, InstanceList testing,
			int[] orders, String defaultLabel,
			String forbidden, String allowed, boolean connected,
			int iterations, double var, HMM hmm) {

		Pattern forbiddenPat = Pattern.compile(forbidden);
		Pattern allowedPat = Pattern.compile(allowed);

		if (hmm == null) {
			hmm = new HMM(training.getPipe(), (Pipe) null);
			
			String startName = hmm.addOrderNStates(training, orders, null,
					defaultLabel, forbiddenPat, allowedPat, connected);
			
			for (int i = 0; i < hmm.numStates(); i++)
				hmm.getState(i).setInitialWeight(Transducer.IMPOSSIBLE_WEIGHT);
			
			hmm.getState(startName).setInitialWeight(0.0);
		}
		
		logger.info("Training on " + training.size() + " instances");
		if (testing != null)
			logger.info("Testing on " + testing.size() + " instances");

		HMMTrainerByLikelihood hmmt = new HMMTrainerByLikelihood(hmm);

		boolean converged;
		for (int i = 1; i <= iterations; i++) {
			converged = hmmt.train(training, 1);

			if (converged)
				break;
		}

		return hmm;
	}

	public static void main(String[] args) throws Exception {
		String corpus = "corpus/train_2.txt";
		String model = "corpus/prueba_hmm.model";

		Reader trainingFile = null;
		trainingFile = new FileReader(new File(corpus));

		Pipe p = null;
		p = new SimpleTaggerSentence2FeatureVectorSequence();
		p.getTargetAlphabet().lookupIndex("O");
		p.setTargetProcessing(true);

		InstanceList trainingData = null;
		trainingData = new InstanceList(p);
		trainingData.addThruPipe(new LineGroupIterator(trainingFile, Pattern
				.compile("^\\s*$"), true));

		logger.info("Number of features in training data: "
				+ p.getDataAlphabet().size());

		if (p.isTargetProcessing()) {
			Alphabet targets = p.getTargetAlphabet();
			StringBuffer buf = new StringBuffer("Labels:");
			for (int i = 0; i < targets.size(); i++)
				buf.append(" ").append(targets.lookupObject(i).toString());
			logger.info(buf.toString());
		}

		int[] list_of_label_Markov_orders = {1};
		int iterations = 500;
		
		HMM hmm = null;
		hmm = train(trainingData, null, list_of_label_Markov_orders, "O",
				"\\s", ".*", true, iterations,
				10, hmm);

		ObjectOutputStream s = new ObjectOutputStream(new FileOutputStream(
				model));
		s.writeObject(hmm);
		s.close();
	}
}
