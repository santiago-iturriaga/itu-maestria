package AII;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.regex.Pattern;

import cc.mallet.fst.CRF;
import cc.mallet.fst.CRFTrainerByL1LabelLikelihood;
import cc.mallet.fst.CRFTrainerByLabelLikelihood;
import cc.mallet.fst.CRFTrainerByStochasticGradient;
import cc.mallet.fst.PerClassAccuracyEvaluator;
import cc.mallet.fst.TokenAccuracyEvaluator;
import cc.mallet.fst.Transducer;
import cc.mallet.fst.TransducerTrainer;
import cc.mallet.pipe.Pipe;
import cc.mallet.pipe.SerialPipes;
import cc.mallet.pipe.SimpleTaggerSentence2TokenSequence;
import cc.mallet.pipe.TokenSequenceLowercase;
import cc.mallet.pipe.iterator.LineGroupIterator;
import cc.mallet.pipe.tsf.FeaturesInWindow;
import cc.mallet.pipe.tsf.OffsetConjunctions;
import cc.mallet.pipe.tsf.OffsetFeatureConjunction;
import cc.mallet.pipe.tsf.RegexMatches;
import cc.mallet.pipe.tsf.TokenFirstPosition;
import cc.mallet.pipe.tsf.TokenTextCharNGrams;
import cc.mallet.pipe.tsf.TokenTextCharPrefix;
import cc.mallet.pipe.tsf.TokenTextCharSuffix;
import cc.mallet.pipe.tsf.TokenTextNGrams;
import cc.mallet.types.Alphabet;
import cc.mallet.types.FeatureVector;
import cc.mallet.types.FeatureVectorSequence;
import cc.mallet.types.Instance;
import cc.mallet.types.InstanceList;
import cc.mallet.types.TokenSequence;

public class FullCRFTrainSimilSVM {

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

				System.out.println(">>>> " + tokens.get(l).getText());
				
				if (tokens.get(l).getFeatures() != null) {
					cc.mallet.util.PropertyList.Iterator iter = tokens.get(l)
							.getFeatures().iterator();
					while (iter.hasNext()) {
						iter.next();

						System.out.print(iter.getKey() + "  ");
						
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
				
				System.out.println("\n");
			}

			carrier.setData(new FeatureVectorSequence(fvs));

			return carrier;
		}
	}

	public static CRF TrainCRF(String trainingFilename) throws IOException {
		ArrayList<Pipe> pipes = new ArrayList<Pipe>();

		pipes.add(new SimpleTaggerSentence2TokenSequence());
		
		// Binary: initial Upper Case, all Upper Case,
		// word: no initial Capital Letter(s), all Lower Case,
		pipes.add(new RegexMatches("CAPITALIZED", Pattern.compile("^\\p{Lu}.*")));
		pipes.add(new TokenFirstPosition("FIRSTTOKEN"));
		
		// Normalizo todo a lowercase
		pipes.add(new TokenSequenceLowercase());
			
		// Word features
		int[][] conjunctionsWords = new int[3][];
		conjunctionsWords[0] = new int[] { -1 };
		conjunctionsWords[1] = new int[] { -2, -1 };
		conjunctionsWords[2] = new int[] { -1, +1 };
		pipes.add(new OffsetConjunctions(conjunctionsWords));
		
//		pipes.add(new TokenTextCharNGrams("BigramChar", new int[] {1, 2}));
//		pipes.add(new TokenTextNGrams("Bigram", new int[] {1, 2}));
		
		// Suffixes
		pipes.add(new TokenTextCharSuffix("S2=", 2));

		// POS features
		pipes.add(new FeaturesInWindow("Suffixes", -1, 0, Pattern.compile("S2=.*"), true));
		
//		pipes.add(new OffsetFeatureConjunction("Suffixes-2", new String[] {"S2=.*"}, new int[] {-2, -1}));
//		pipes.add(new OffsetFeatureConjunction("Suffixes-1", new String[] {"S2=.*"}, new int[] {-1}));
		
		// Preffixes
//		pipes.add(new TokenTextCharPrefix("P2=", 2));
				
		// features: contains a (period / number / hyphen ...)	
		pipes.add(new RegexMatches("STARTSNUMBER", Pattern.compile("^[0-9].*")));
		pipes.add(new RegexMatches("NUMBER", Pattern.compile(".*[0-9].*")));
		pipes.add(new RegexMatches("HYPHENATED", Pattern
				.compile(".*[\\-|\\_].*")));
		pipes.add(new RegexMatches("DOLLARSIGN", Pattern.compile(".*\\$.*")));
		pipes.add(new RegexMatches("SIGN", Pattern
				.compile(".*[\\.|,|\"|:|;].*")));		
		
		// Punctuation: punctuation (’.’, ’ ?’, ’ !’)
		pipes.add(new RegexMatches("SIGN-QUESTION", Pattern.compile(".*\\?.*")));
		pipes.add(new RegexMatches("SIGN-EXCLAMATION", Pattern.compile(".*\\!.*")));
		pipes.add(new RegexMatches("SIGN-END", Pattern.compile(".*\\..*")));
				
		pipes.add(new FullCRFTrainSimilSVM.SimpleTokenSentence2FeatureVectorSequence());
		Pipe pipe = new SerialPipes(pipes);

		InstanceList trainingInstances = new InstanceList(pipe);
		trainingInstances.addThruPipe(new LineGroupIterator(new BufferedReader(
				new InputStreamReader(new FileInputStream(trainingFilename))),
				Pattern.compile("^\\s*$"), true));
		
		CRF crf = new CRF(pipe, null);
		
		int[] orders = { 1 };
		Pattern forbiddenPat = Pattern.compile("\\s");
		Pattern allowedPat = Pattern.compile(".*");

		String startName = crf.addOrderNStates(trainingInstances, orders,
				null, "O", forbiddenPat, allowedPat, true);
		for (int s = 0; s < crf.numStates(); s++)
			crf.getState(s).setInitialWeight(Transducer.IMPOSSIBLE_WEIGHT);
		crf.getState(startName).setInitialWeight(0.0);

		TransducerTrainer trainer = null;
		trainer = new CRFTrainerByLabelLikelihood(crf);
		((CRFTrainerByLabelLikelihood) trainer).setGaussianPriorVariance(10.0);
		
		trainer.train(trainingInstances, 500);
		
		return crf;
	}

	public static void main(String[] args) throws Exception {
		int i = 2;
//		for (int i = 0; i < 10; i++) {
			String train = "corpus/train_" + i + ".txt";
//			String test = "corpus/test_full_" + i + ".txt";
			String model = "model_crf/svm_crf_" + i + ".model";

			CRF modelObj = TrainCRF(train);

			ObjectOutputStream s = new ObjectOutputStream(
					new FileOutputStream(model));
			s.writeObject(modelObj);
			s.close();
//		}
	}

}
