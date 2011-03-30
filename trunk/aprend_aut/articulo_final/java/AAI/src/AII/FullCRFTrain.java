package AII;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.ObjectOutputStream;
import java.io.PrintWriter;
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
import cc.mallet.pipe.tsf.OffsetFeatureConjunction;
import cc.mallet.pipe.tsf.RegexMatches;
import cc.mallet.pipe.tsf.TokenFirstPosition;
import cc.mallet.pipe.tsf.TokenText;
import cc.mallet.types.Alphabet;
import cc.mallet.types.FeatureVector;
import cc.mallet.types.FeatureVectorSequence;
import cc.mallet.types.Instance;
import cc.mallet.types.InstanceList;
import cc.mallet.types.TokenSequence;

public class FullCRFTrain {

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

	public static CRF TrainCRF(String trainingFilename)
			throws IOException {
		ArrayList<Pipe> pipes = new ArrayList<Pipe>();

		pipes.add(new SimpleTaggerSentence2TokenSequence());
		pipes.add(new RegexMatches("CAPITALIZED", Pattern.compile("^\\p{Lu}.*")));
		pipes.add(new RegexMatches("SIGN-PUNCT", Pattern
				.compile("(,|-|:|;|\\.|\\*0\\*)")));
		pipes.add(new RegexMatches("SIGN-QE", Pattern.compile("(\\?|¿|!|¡)")));
		pipes.add(new RegexMatches("SIGN-ALL", Pattern
				.compile("(,|-|:|;|\\.|\\*0\\*|\\?|¿|!|¡|\")")));
		
		pipes.add(new RegexMatches("CP-1", Pattern.compile("(por|-|para|en|sobre|ver|a|saber|sé|¿)")));
		pipes.add(new RegexMatches("CP-2", Pattern.compile("(no|¿|para|-|se|que)")));
		pipes.add(new RegexMatches("CP+1", Pattern.compile("(\\?|no|,|es|le|significa)")));
		
		pipes.add(new RegexMatches("ADVERBIO", Pattern
				.compile("(cuando|cuanto|donde|que|como|adonde)")));

		pipes.add(new TokenFirstPosition("FIRST"));
		pipes.add(new TokenSequenceLowercase());
		pipes.add(new TokenText("WORD="));

		pipes.add(new OffsetFeatureConjunction("PREV-FIRST",
				new String[] { "SIGN-ALL" }, new int[] { -1 }));

		pipes.add(new OffsetFeatureConjunction("SECOND",
				new String[] { "FIRST" }, new int[] { -1 }));

		pipes.add(new OffsetFeatureConjunction("PREV-ADVERBIO",
				new String[] { "ADVERBIO" }, new int[] { -1 }));

		pipes.add(new OffsetFeatureConjunction("REGLA-1", new String[] {
				"CP-1", "ADVERBIO" }, new int[] { -1, 0 }));
		
		pipes.add(new OffsetFeatureConjunction("REGLA-2", new String[] {
				"CP-2", "ADVERBIO" }, new int[] { -2, 0 }));
		
		pipes.add(new OffsetFeatureConjunction("REGLA+1", new String[] {
				"ADVERBIO", "CP+1" }, new int[] { 0, 1 }));

		// pipes.add(new TokenTextCharSuffix("S4=", 4));
		// pipes.add(new TokenTextCharSuffix("S3=", 3));
		// pipes.add(new TokenTextCharSuffix("S2=", 2));
		pipes.add(new CRFTrainFinal.SimpleTokenSentence2FeatureVectorSequence());

		Pipe pipe = new SerialPipes(pipes);

		InstanceList trainingInstances = new InstanceList(pipe);
		trainingInstances.addThruPipe(new LineGroupIterator(new BufferedReader(
				new InputStreamReader(new FileInputStream(trainingFilename))),
				Pattern.compile("^\\s*$"), true));

		CRF crf = new CRF(pipe, null);

		int[] orders = { 1 };
		Pattern forbiddenPat = Pattern.compile("(CON_TILDE,CON_TILDE)");
		Pattern allowedPat = Pattern.compile(".*");

		String startName = crf.addOrderNStates(trainingInstances, orders, null,
				"O", forbiddenPat, allowedPat, true);
		for (int s = 0; s < crf.numStates(); s++)
			crf.getState(s).setInitialWeight(Transducer.IMPOSSIBLE_WEIGHT);
		crf.getState(startName).setInitialWeight(0.0);

		CRFTrainerByLabelLikelihood trainer = null;

		trainer = new CRFTrainerByLabelLikelihood(crf);
		trainer.setGaussianPriorVariance(11.0);
		trainer.setUseSparseWeights(true);
		trainer.setUseSomeUnsupportedTrick(true);

		trainer.train(trainingInstances, 1000);

		return crf;
	}

	public static void main(String[] args) throws Exception {
		for (int i = 0; i < 10; i++) {
			String train = "corpus/train_" + i + ".txt";
			String model = "model_crf/f_" + i + ".model";

			CRF modelObj = TrainCRF(train);

			ObjectOutputStream s = new ObjectOutputStream(new FileOutputStream(
					model));
			s.writeObject(modelObj);
			s.close();
		}
	}

}
