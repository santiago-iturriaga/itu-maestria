package AII;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.regex.Pattern;

import cc.mallet.fst.CRF;
import cc.mallet.fst.CRFTrainerByLabelLikelihood;
import cc.mallet.fst.PerClassAccuracyEvaluator;
import cc.mallet.fst.TokenAccuracyEvaluator;
import cc.mallet.fst.Transducer;
import cc.mallet.pipe.Pipe;
import cc.mallet.pipe.SerialPipes;
import cc.mallet.pipe.SimpleTaggerSentence2TokenSequence;
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

	public static CRF TrainCRF(String trainingFilename) throws IOException {
		ArrayList<Pipe> pipes = new ArrayList<Pipe>();

		pipes.add(new SimpleTaggerSentence2TokenSequence());
		pipes.add(new RegexMatches("CAPITALIZED", Pattern.compile("^\\p{Lu}.*")));
		pipes.add(new RegexMatches("STARTSNUMBER", Pattern.compile("^[0-9].*")));
		pipes.add(new RegexMatches("HYPHENATED", Pattern.compile(".*[\\-|\\_].*")));
		pipes.add(new RegexMatches("DOLLARSIGN", Pattern.compile(".*\\$.*")));
		pipes.add(new TokenFirstPosition("FIRSTTOKEN"));
		pipes.add(new TokenSequenceLowercase());
		pipes.add(new CRFTrain.SimpleTokenSentence2FeatureVectorSequence());
		Pipe pipe = new SerialPipes(pipes);

		InstanceList trainingInstances = new InstanceList(pipe);
		trainingInstances.addThruPipe(new LineGroupIterator(new BufferedReader(
				new InputStreamReader(new FileInputStream(trainingFilename))),
				Pattern.compile("^\\s*$"), true));

		CRF crf = new CRF(pipe, null);

		// Errores: 3/0/0
		int[] orders = { 1 };
		Pattern forbiddenPat = Pattern.compile("\\s");
		Pattern allowedPat = Pattern.compile(".*");

		String startName = crf.addOrderNStates(trainingInstances, orders, null,
				"O", forbiddenPat, allowedPat, true);
		for (int i = 0; i < crf.numStates(); i++)
			crf.getState(i).setInitialWeight(Transducer.IMPOSSIBLE_WEIGHT);
		crf.getState(startName).setInitialWeight(0.0);
		
//		crf.addStatesForLabelsConnectedAsIn(trainingInstances);
//		crf.addStartState();
		
		// Muchos errores
//		crf.addStatesForBiLabelsConnectedAsIn(trainingInstances);
//		crf.addStartState();
		
		// Errores: 3/0/2
//		crf.addStatesForHalfLabelsConnectedAsIn(trainingInstances);
//		crf.addStartState();
		
		// Errores: 3/0/2
//		crf.addStatesForThreeQuarterLabelsConnectedAsIn(trainingInstances);
//		crf.addStartState();
	
		CRFTrainerByLabelLikelihood trainer = new CRFTrainerByLabelLikelihood(
				crf);
		trainer.setGaussianPriorVariance(10.0);

		// CRFTrainerByStochasticGradient trainer =
		// new CRFTrainerByStochasticGradient(crf, 1.0);

		// CRFTrainerByL1LabelLikelihood trainer =
		// new CRFTrainerByL1LabelLikelihood(crf, 0.75);

		// trainer.addEvaluator(new PerClassAccuracyEvaluator(trainingInstances,
		// "training"));
		trainer.train(trainingInstances, 500);

		return crf;
	}

	public static void main(String[] args) throws Exception {
		String train = "corpus/train_2.txt";
		String test = "corpus/test_2.txt";
		String model = "corpus/crf_2.model";

		CRF crf = null;
		crf = TrainCRF(train);

		ObjectOutputStream s = new ObjectOutputStream(new FileOutputStream(
				model));
		s.writeObject(crf);
		s.close();
	}
}
