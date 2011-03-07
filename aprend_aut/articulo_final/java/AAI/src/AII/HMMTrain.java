package AII;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.regex.Pattern;

import cc.mallet.fst.HMM;
import cc.mallet.fst.HMMTrainerByLikelihood;
import cc.mallet.fst.PerClassAccuracyEvaluator;
import cc.mallet.fst.Transducer;
import cc.mallet.fst.TransducerEvaluator;
import cc.mallet.pipe.Pipe;
import cc.mallet.pipe.SerialPipes;
import cc.mallet.pipe.SimpleTaggerSentence2TokenSequence;
import cc.mallet.pipe.TokenSequence2FeatureSequence;
import cc.mallet.pipe.TokenSequenceLowercase;
import cc.mallet.pipe.iterator.LineGroupIterator;
import cc.mallet.pipe.tsf.OffsetConjunctions;
import cc.mallet.pipe.tsf.RegexMatches;
import cc.mallet.pipe.tsf.TokenFirstPosition;
import cc.mallet.types.Alphabet;
import cc.mallet.types.FeatureVector;
import cc.mallet.types.FeatureVectorSequence;
import cc.mallet.types.Instance;
import cc.mallet.types.InstanceList;
import cc.mallet.types.TokenSequence;

public class HMMTrain {
	public static class SimpleTokenSentence2FeatureSequence extends Pipe {
		private static final long serialVersionUID = -2059308802200728626L;

		public SimpleTokenSentence2FeatureSequence(Alphabet dataDict) {
			super(dataDict, null);
		}

		public SimpleTokenSentence2FeatureSequence() {
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
	
	public static HMM TrainHMM(String trainingFilename) throws IOException {
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
		
		HMM hmm = new HMM(pipe, null);

		int[] orders = { 1 };
	    Pattern forbiddenPat = Pattern.compile("\\s");
	    Pattern allowedPat = Pattern.compile(".*");
		
		String startName = hmm.addOrderNStates(trainingInstances, orders, null,
				"O",forbiddenPat, allowedPat, true);
		for (int i = 0; i < hmm.numStates(); i++)
			hmm.getState(i).setInitialWeight(Transducer.IMPOSSIBLE_WEIGHT);
		hmm.getState(startName).setInitialWeight(0.0);
		
		//hmm.addStatesForLabelsConnectedAsIn(trainingInstances);
		
		HMMTrainerByLikelihood trainer = 
			new HMMTrainerByLikelihood(hmm);
		
		trainer.train(trainingInstances, 500);
		
		return hmm;
	}

	public static void main(String[] args) throws Exception {
		String train = "corpus/train_2.txt";
		String test = "corpus/test_2.txt";
		String model = "corpus/hmm_2.model";

		HMM hmm = TrainHMM(train);

		ObjectOutputStream s = new ObjectOutputStream(new FileOutputStream(
				model));
		s.writeObject(hmm);
		s.close();
	}
}
