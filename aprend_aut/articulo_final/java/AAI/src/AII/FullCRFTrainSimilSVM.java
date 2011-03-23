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
import cc.mallet.fst.Transducer;
import cc.mallet.fst.TransducerTrainer;
import cc.mallet.pipe.Pipe;
import cc.mallet.pipe.SerialPipes;
import cc.mallet.pipe.SimpleTaggerSentence2TokenSequence;
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
		
		pipes.add(new RegexMatches("CAPITALIZED", Pattern.compile("^\\p{Lu}.*")));
		pipes.add(new RegexMatches("STARTSNUMBER", Pattern.compile("^[0-9].*")));
		pipes.add(new RegexMatches("HYPHENATED", Pattern
				.compile(".*[\\-|\\_].*")));
		pipes.add(new RegexMatches("DOLLARSIGN", Pattern.compile(".*\\$.*")));
		pipes.add(new RegexMatches("SIGN", Pattern.compile(".*[\\!|\\?].*")));
		pipes.add(new TokenFirstPosition("FIRSTTOKEN"));
		pipes.add(new TokenSequenceLowercase());
		
		//pipes.add(new FeaturesInWindow("PREV-", -1, 1));
		//pipes.add(new FeaturesInWindow("NEXT-", 1, 2));
		
		// Word features: w−3 , w−2 , w−1 , w0 , w+1, w+2 , w+3
		// Word bigrams: (w−2 , w−1 ), (w−1 , w+1), (w−1 , w0 ), (w0 , w+1 ), (w+1 , w+2)
		// Word trigrams: (w−2 , w−1 , w0 ), (w−2, w−1 , w+1 ),
		//	(w−1 , w0 , w+1 ), (w−1, w+1 , w+2 ), (w0 , w+1 , w+2 )
		
		// POS features: p−3 , p−2 , p−1 , p0 , p+1 , p+2 , p+3
		// POS bigrams: (p−2 , p−1 ), (p−1 , a+1 ), (a+1 , a+2 )
		// POS trigrams: (p−2 , p−1 , a+0 ), (p−2, p−1 , a+1 ),
		//	(p−1 , a0 , a+1 ), (p−1 , a+1 , a+2 )
		
		// Ambiguity class: a0 , a1 , a2 , a3
		// may_be's: m0 , m1 , m2 , m3

		// Punctuation: punctuation (’.’, ’ ?’, ’ !’)
		// Suffixes: s1 , s1 s2 , s1 s2 s3 , s1 s2 s3 s4
		// Preffixes: sn , sn-1 sn , sn-2 sn-1 sn , sn-3 sn-2 sn-1 sn

		// Binary: initial Upper Case, all Upper Case,
		// word: no initial Capital Letter(s), all Lower Case,
		// features: contains a (period / number / hyphen ...)
		// word length: integer	
		
		pipes.add(new FullCRFTrainSimilSVM.SimpleTokenSentence2FeatureVectorSequence());
		Pipe pipe = new SerialPipes(pipes);

		InstanceList trainingInstances = new InstanceList(pipe);
		trainingInstances.addThruPipe(new LineGroupIterator(new BufferedReader(
				new InputStreamReader(new FileInputStream(trainingFilename))),
				Pattern.compile("^\\s*$"), true));

		CRF crf = new CRF(pipe, null);
		
//		crf.addFullyConnectedStatesForThreeQuarterLabels(trainingInstances);
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
		for (int i = 0; i < 10; i++) {
			String train = "corpus/train_" + i + ".txt";
			String model = "model_crf/crf_" + i + ".model";

			CRF modelObj = TrainCRF(train);

			ObjectOutputStream s = new ObjectOutputStream(
					new FileOutputStream(model));
			s.writeObject(modelObj);
			s.close();
		}
	}

}
