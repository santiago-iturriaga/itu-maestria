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
import cc.mallet.fst.InstanceAccuracyEvaluator;
import cc.mallet.fst.PerClassAccuracyEvaluator;
import cc.mallet.fst.TokenAccuracyEvaluator;
import cc.mallet.fst.Transducer;
import cc.mallet.fst.TransducerEvaluator;
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
import cc.mallet.types.AugmentableFeatureVector;
import cc.mallet.types.FeatureVector;
import cc.mallet.types.FeatureVectorSequence;
import cc.mallet.types.Instance;
import cc.mallet.types.InstanceList;
import cc.mallet.types.TokenSequence;

public class CRFTrainInduce {

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

//				System.out.println(">>>> " + tokens.get(l).getText());
				
				if (tokens.get(l).getFeatures() != null) {
					cc.mallet.util.PropertyList.Iterator iter = tokens.get(l)
							.getFeatures().iterator();
					while (iter.hasNext()) {
						iter.next();

//						System.out.print(iter.getKey() + "  ");
						
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

				fvs[l] = new AugmentableFeatureVector(features, featureIndicesArr, null, featureIndicesArr.length);
				
//				System.out.println("\n");
			}

			carrier.setData(new FeatureVectorSequence(fvs));

			return carrier;
		}
	}

	public static CRF TrainCRF(String trainingFilename, String testingFilename) throws IOException {
		ArrayList<Pipe> pipes = new ArrayList<Pipe>();

		pipes.add(new SimpleTaggerSentence2TokenSequence());
		
		// Binary: initial Upper Case, all Upper Case,
		// word: no initial Capital Letter(s), all Lower Case,
		pipes.add(new RegexMatches("CAPITALIZED", Pattern.compile("^\\p{Lu}.*")));
		pipes.add(new TokenFirstPosition("FIRSTTOKEN"));
		
		// Normalizo todo a lowercase
		pipes.add(new TokenSequenceLowercase());
			
		// Suffixes
		pipes.add(new TokenTextCharSuffix("S2=", 2));
		pipes.add(new TokenTextCharSuffix("S3=", 3));

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
				
		pipes.add(new CRFTrainInduce.SimpleTokenSentence2FeatureVectorSequence());
		Pipe pipe = new SerialPipes(pipes);

		InstanceList trainingInstances = new InstanceList(pipe);
		trainingInstances.addThruPipe(new LineGroupIterator(new BufferedReader(
				new InputStreamReader(new FileInputStream(trainingFilename))),
				Pattern.compile("^\\s*$"), true));
		
		InstanceList testingInstances = new InstanceList(pipe);
		testingInstances.addThruPipe(new LineGroupIterator(new BufferedReader(
				new InputStreamReader(new FileInputStream(testingFilename))),
				Pattern.compile("^\\s*$"), true));
		
		CRF crf = new CRF(pipe, null);
		crf.addFullyConnectedStatesForLabels();

		CRFTrainerByLabelLikelihood trainer = null;
		trainer = new CRFTrainerByLabelLikelihood(crf);
		trainer.setGaussianPriorVariance(10.0);
	
//	      if (weightsOption.value.equals("dense")) {
//	          crft.setUseSparseWeights(false);
//	          crft.setUseSomeUnsupportedTrick(false);
//	        }
//	        else if (weightsOption.value.equals("some-dense")) {
//	          crft.setUseSparseWeights(true);
//	          crft.setUseSomeUnsupportedTrick(true);
//	        }
//	        else if (weightsOption.value.equals("sparse")) {
//	          crft.setUseSparseWeights(true);
//	          crft.setUseSomeUnsupportedTrick(false);
//	        }
		
//		trainer.addEvaluator(new InstanceAccuracyEvaluator());
//		trainer.addEvaluator(new PerClassAccuracyEvaluator(testingInstances, "testing"));
//		trainer.addEvaluator(new TokenAccuracyEvaluator(testingInstances, "testing"));
				
		TransducerEvaluator eval = new TokenAccuracyEvaluator(new InstanceList[] {trainingInstances, testingInstances}, new String[] {"Training", "Testing"});
		
		trainer.trainWithFeatureInduction(trainingInstances, null, testingInstances, eval, 1000, 10, 20, 500, 0.5, false, null);
//		trainer.train(trainingInstances, 1000);
		
		return crf;
	}

	public static void main(String[] args) throws Exception {
		int i = 2;
//		for (int i = 0; i < 10; i++) {
			String train = "corpus/train_" + i + ".txt";
			String test = "corpus/test_full_" + i + ".txt";
			String model = "model_crf/induce_crf_" + i + ".model";

			CRF modelObj = TrainCRF(train, test);

			ObjectOutputStream s = new ObjectOutputStream(
					new FileOutputStream(model));
			s.writeObject(modelObj);
			s.close();
//		}
	}

}
