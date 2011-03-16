package AII;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.regex.Pattern;

import cc.mallet.fst.CRF;
import cc.mallet.fst.CRFTrainerByL1LabelLikelihood;
import cc.mallet.fst.CRFTrainerByLabelLikelihood;
import cc.mallet.fst.CRFTrainerByStochasticGradient;
import cc.mallet.fst.HMM;
import cc.mallet.fst.PerClassAccuracyEvaluator;
import cc.mallet.fst.TokenAccuracyEvaluator;
import cc.mallet.fst.Transducer;
import cc.mallet.fst.TransducerTrainer;
import cc.mallet.pipe.Pipe;
import cc.mallet.pipe.SerialPipes;
import cc.mallet.pipe.SimpleTaggerSentence2TokenSequence;
import cc.mallet.pipe.TokenSequence2FeatureSequence;
import cc.mallet.pipe.TokenSequence2FeatureVectorSequence;
import cc.mallet.pipe.TokenSequenceLowercase;
import cc.mallet.pipe.iterator.LineGroupIterator;
import cc.mallet.pipe.tsf.FeaturesInWindow;
import cc.mallet.pipe.tsf.OffsetConjunctions;
import cc.mallet.pipe.tsf.RegexMatches;
import cc.mallet.pipe.tsf.TokenFirstPosition;
import cc.mallet.pipe.tsf.TokenTextCharSuffix;
import cc.mallet.types.Alphabet;
import cc.mallet.types.AugmentableFeatureVector;
import cc.mallet.types.FeatureVector;
import cc.mallet.types.FeatureVectorSequence;
import cc.mallet.types.Instance;
import cc.mallet.types.InstanceList;
import cc.mallet.types.LabelAlphabet;
import cc.mallet.types.LabelSequence;
import cc.mallet.types.Token;
import cc.mallet.types.TokenSequence;
import cc.mallet.util.PropertyList;

public class CRFTrain {

	private CRFTrain() {
	}

	public static class SimpleTokenSentence2FeatureVectorSequence extends Pipe {
		private static final long serialVersionUID = -2059308802200728626L;

		public SimpleTokenSentence2FeatureVectorSequence(Alphabet dataDict) {
			super(dataDict, null);
		}

		public SimpleTokenSentence2FeatureVectorSequence() {
			super(new Alphabet(), null);
		}

		public Instance pipe(Instance carrier) {
			TokenSequence tokens = (TokenSequence) carrier.getData();
			Alphabet features = getDataAlphabet();

			FeatureVector[] fvs = new FeatureVector[tokens.size()];

			for (int l = 0; l < tokens.size(); l++) {
				ArrayList<Integer> featureIndices = new ArrayList<Integer>();

				int featureIndex;
				featureIndex = features.lookupIndex(tokens.get(l).getText());
				if (featureIndex >= 0) {
					featureIndices.add(featureIndex);
				}

				if (tokens.get(l).getFeatures() != null) {
					cc.mallet.util.PropertyList.Iterator iter = tokens.get(l)
							.getFeatures().iterator();
					while (iter.hasNext()) {
						iter.next();

						if (iter.hasNext() || !isTargetProcessing()) {
							featureIndex = features.lookupIndex(iter.getKey());
							if (featureIndex >= 0) {
								featureIndices.add(featureIndex);
							}
						}
					}
				}

				int[] featureIndicesArr = new int[featureIndices.size()];
				for (int index = 0; index < featureIndices.size(); index++) {
					featureIndicesArr[index] = featureIndices.get(index);
				}

				fvs[l] = new FeatureVector(features, featureIndicesArr);
			}

			carrier.setData(new FeatureVectorSequence(fvs));

			return carrier;
		}
	}

	public static CRF TrainCRF(String trainingFilename, int i, int p, int t)
			throws IOException {
		ArrayList<Pipe> pipes = new ArrayList<Pipe>();

		pipes.add(new SimpleTaggerSentence2TokenSequence());
		pipes.add(new RegexMatches("CAPITALIZED", Pattern.compile("^\\p{Lu}.*")));
		pipes.add(new RegexMatches("STARTSNUMBER", Pattern.compile("^[0-9].*")));
		pipes.add(new RegexMatches("HYPHENATED", Pattern
				.compile(".*[\\-|\\_].*")));
		pipes.add(new RegexMatches("DOLLARSIGN", Pattern.compile(".*\\$.*")));
		pipes.add(new TokenFirstPosition("FIRSTTOKEN"));
		pipes.add(new TokenSequenceLowercase());

		if (p == 0) {
			int[][] conjunctions = new int[1][];
			conjunctions[0] = new int[] { 1 };
			pipes.add(new OffsetConjunctions(conjunctions));
			pipes.add(new CRFTrain.SimpleTokenSentence2FeatureVectorSequence());
		} else if (p == 1) {
			int[][] conjunctions = new int[2][];
			conjunctions[0] = new int[] { 1 };
			conjunctions[1] = new int[] { -1 };
			pipes.add(new OffsetConjunctions(conjunctions));
			pipes.add(new CRFTrain.SimpleTokenSentence2FeatureVectorSequence());
		} else if (p == 2) {
			pipes.add(new CRFTrain.SimpleTokenSentence2FeatureVectorSequence());
		}

		Pipe pipe = new SerialPipes(pipes);

		InstanceList trainingInstances = new InstanceList(pipe);
		trainingInstances.addThruPipe(new LineGroupIterator(new BufferedReader(
				new InputStreamReader(new FileInputStream(trainingFilename))),
				Pattern.compile("^\\s*$"), true));

		CRF crf = new CRF(pipe, null);

		if (i == 0)
			crf.addFullyConnectedStatesForBiLabels();
		else if (i == 1)
			crf.addFullyConnectedStatesForLabels();
		else if (i == 2)
			crf.addFullyConnectedStatesForThreeQuarterLabels(trainingInstances);
		else if (i == 3)
			crf.addFullyConnectedStatesForTriLabels();
		else if (i == 4)
			crf.addStatesForBiLabelsConnectedAsIn(trainingInstances);
		else if (i == 5)
			crf.addStatesForHalfLabelsConnectedAsIn(trainingInstances);
		else if (i == 6)
			crf.addStatesForLabelsConnectedAsIn(trainingInstances);
		else if (i == 7)
			crf.addStatesForThreeQuarterLabelsConnectedAsIn(trainingInstances);
		else if (i == 8) {
			int[] orders = { 1 };
			Pattern forbiddenPat = Pattern.compile("\\s");
			Pattern allowedPat = Pattern.compile(".*");

			String startName = crf.addOrderNStates(trainingInstances, orders,
					null, "O", forbiddenPat, allowedPat, true);
			for (int s = 0; s < crf.numStates(); s++)
				crf.getState(s).setInitialWeight(Transducer.IMPOSSIBLE_WEIGHT);
			crf.getState(startName).setInitialWeight(0.0);
		}

		TransducerTrainer trainer = null;

		if (t == 0) {
			trainer = new CRFTrainerByLabelLikelihood(crf);
			((CRFTrainerByLabelLikelihood) trainer)
					.setGaussianPriorVariance(10.0);
		} else if (t == 1) {
			trainer = new CRFTrainerByStochasticGradient(crf, 1.0);
		} else if (t == 2) {
			trainer = new CRFTrainerByL1LabelLikelihood(crf, 0.75);
		}

		trainer.train(trainingInstances, 500);

		return crf;
	}

	public static void main(String[] args) throws Exception {
		String train = "corpus/train_2.txt";

		for (int t = 0; t < 3; t++) {
			for (int p = 0; p < 3; p++) {
				for (int i = 0; i < 9; i++) {
					String model = "corpus/crf_" + i + "_" + p + "_" + t
							+ ".model";

					File modelFile = new File(model);
					if (!modelFile.exists()) {
						CRF modelObj = TrainCRF(train, i, p, t);

						ObjectOutputStream s = new ObjectOutputStream(
								new FileOutputStream(model));
						s.writeObject(modelObj);
						s.close();
					}
				}
			}
		}
	}
}
