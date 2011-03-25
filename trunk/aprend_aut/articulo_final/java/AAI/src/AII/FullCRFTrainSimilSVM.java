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
import cc.mallet.pipe.tsf.TokenTextCharSuffix;
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
		
		// Binary: initial Upper Case, all Upper Case,
		// word: no initial Capital Letter(s), all Lower Case,
		pipes.add(new RegexMatches("CAPITALIZED", Pattern.compile("^\\p{Lu}.*")));
		pipes.add(new TokenFirstPosition("FIRSTTOKEN"));
		
		// Normalizo todo a lowercase
		pipes.add(new TokenSequenceLowercase());
			
		// Word features: 	w−3 , w−2 , w−1 , w0 , w+1, w+2 , w+3
		// Word bigrams: 	(w−2 , w−1 ), (w−1 , w+1), (w−1 , w0 ), (w0 , w+1 ), (w+1 , w+2)
		// Word trigrams: 	(w−2 , w−1 , w0 ), (w−2, w−1 , w+1 ),
		//					(w−1 , w0 , w+1 ), (w−1, w+1 , w+2 ), (w0 , w+1 , w+2 )
//		int[][] conjunctionsWords = new int[12][];
//		conjunctionsWords[0] = new int[] { -3 };
//		conjunctionsWords[1] = new int[] { -2 };
//		conjunctionsWords[2] = new int[] { -1 };
//		conjunctionsWords[3] = new int[] { 0 };
//		conjunctionsWords[4] = new int[] { 1 };
//		conjunctionsWords[5] = new int[] { 2 };
//		conjunctionsWords[6] = new int[] { 3 };
//		conjunctionsWords[7] = new int[] { -2, -1 };
//		conjunctionsWords[8] = new int[] { -1, 1 };
//		conjunctionsWords[9] = new int[] { -1, 0 };
//		conjunctionsWords[10] = new int[] { 0, 1 };
//		conjunctionsWords[11] = new int[] { 1, 2 };
//		conjunctionsWords[12] = new int[] { -2, -1, 0 };
//		conjunctionsWords[13] = new int[] { -2, -1, 1 };
//		conjunctionsWords[14] = new int[] { -1, 0, 1 };
//		conjunctionsWords[15] = new int[] { -1, 1, 2 };
//		conjunctionsWords[16] = new int[] { 0, 1, 2 };
//		pipes.add(new OffsetConjunctions(conjunctionsWords));
		
		// POS features:	p−3 , p−2 , p−1 , p0 , p+1 , p+2 , p+3
		// POS bigrams:		(p−2 , p−1 ), (p−1 , a+1 ), (a+1 , a+2 )
		// POS trigrams:	(p−2 , p−1 , a+0 ), (p−2, p−1 , a+1 ),
		//					(p−1 , a0 , a+1 ), (p−1 , a+1 , a+2 )
//		pipes.add(new FeaturesInWindow("P-3-", -3, -3));
//		pipes.add(new FeaturesInWindow("P-2-", -2, -2));
//		pipes.add(new FeaturesInWindow("P-1-", -1, -1));
//		pipes.add(new FeaturesInWindow("P-2-1-", -2, -1));
				
		// Ambiguity class: a0 , a1 , a2 , a3
		// may_be's: m0 , m1 , m2 , m3

		// Suffixes: s1 , s1 s2 , s1 s2 s3 , s1 s2 s3 s4
		pipes.add(new TokenTextCharSuffix("S1=", 1));
		pipes.add(new TokenTextCharSuffix("S2=", 2));
		pipes.add(new TokenTextCharSuffix("S3=", 3));
		pipes.add(new TokenTextCharSuffix("S4=", 4));
		
		// Preffixes: sn , sn-1 sn , sn-2 sn-1 sn , sn-3 sn-2 sn-1 sn
		pipes.add(new TokenTextCharSuffix("P1=", 1));
		pipes.add(new TokenTextCharSuffix("P2=", 2));
		pipes.add(new TokenTextCharSuffix("P3=", 3));
		pipes.add(new TokenTextCharSuffix("P4=", 4));
				
		// features: contains a (period / number / hyphen ...)	
		pipes.add(new RegexMatches("STARTSNUMBER", Pattern.compile("^[0-9].*")));
		pipes.add(new RegexMatches("NUMBER", Pattern.compile(".*[0-9].*")));
		pipes.add(new RegexMatches("HYPHENATED", Pattern
				.compile(".*[\\-|\\_].*")));
		pipes.add(new RegexMatches("DOLLARSIGN", Pattern.compile(".*\\$.*")));

		// Punctuation: punctuation (’.’, ’ ?’, ’ !’)
		pipes.add(new RegexMatches("SIGN-QUESTION", Pattern.compile(".*\\?.*")));
		pipes.add(new RegexMatches("SIGN-EXCLAMATION", Pattern.compile(".*\\!.*")));
		pipes.add(new RegexMatches("SIGN-END", Pattern.compile(".*\\..*")));
		
		// word length: integer
		
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
//		int i = 0;
		for (int i = 0; i < 10; i++) {
			String train = "corpus/train_" + i + ".txt";
//			String test = "corpus/test_full_" + i + ".txt";
			String model = "model_crf/crf_" + i + ".model";

			CRF modelObj = TrainCRF(train);

			ObjectOutputStream s = new ObjectOutputStream(
					new FileOutputStream(model));
			s.writeObject(modelObj);
			s.close();
		}
	}

}
