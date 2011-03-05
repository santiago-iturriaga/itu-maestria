package AII;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
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

public class CRFTrain {

	private CRFTrain() {
	}

	public static CRF TrainCRF(String trainingFilename)
			throws IOException {
		ArrayList<Pipe> pipes = new ArrayList<Pipe>();

		int[][] conjunctions = new int[2][];
		conjunctions[0] = new int[] { -1 };
		conjunctions[1] = new int[] { 1 };

		pipes.add(new SimpleTaggerSentence2TokenSequence(true));
//		pipes.add(new OffsetConjunctions(conjunctions));
		// pipes.add(new FeaturesInWindow("PREV-", -1, 1));
		// pipes.add(new TokenTextCharSuffix("C1=", 1));
		// pipes.add(new TokenTextCharSuffix("C2=", 2));
		// pipes.add(new TokenTextCharSuffix("C3=", 3));
//		pipes.add(new RegexMatches("CAPITALIZED", Pattern.compile("^\\p{Lu}.*")));
//		pipes.add(new RegexMatches("STARTSNUMBER", Pattern.compile("^[0-9].*")));
//		pipes.add(new RegexMatches("HYPHENATED", Pattern.compile(".*[\\-|_].*")));
		// pipes.add(new RegexMatches("DOLLARSIGN", Pattern.compile(".*\\$.*")));
//		pipes.add(new TokenFirstPosition("FIRSTTOKEN"));
//		pipes.add(new TokenSequenceLowercase());
		pipes.add(new TokenSequence2FeatureVectorSequence());

		Pipe pipe = new SerialPipes(pipes);

		InstanceList trainingInstances = new InstanceList(pipe);
		trainingInstances.addThruPipe(new LineGroupIterator(new BufferedReader(
				new InputStreamReader(new FileInputStream(trainingFilename))),
				Pattern.compile("^\\s*$"), true));

		CRF crf = new CRF(pipe, null);

		int[] orders = { 1 };
	    Pattern forbiddenPat = Pattern.compile("\\s");
	    Pattern allowedPat = Pattern.compile(".*");
		
		String startName = crf.addOrderNStates(trainingInstances, orders, null,
				"O",forbiddenPat, allowedPat, true);
		for (int i = 0; i < crf.numStates(); i++)
			crf.getState(i).setInitialWeight(Transducer.IMPOSSIBLE_WEIGHT);
		crf.getState(startName).setInitialWeight(0.0);

//	    crf.addStatesForLabelsConnectedAsIn(trainingInstances);
	    
		CRFTrainerByLabelLikelihood trainer = new CRFTrainerByLabelLikelihood(
				crf);
		trainer.setGaussianPriorVariance(10.0);

		// CRFTrainerByStochasticGradient trainer =
		// new CRFTrainerByStochasticGradient(crf, 1.0);

		// CRFTrainerByL1LabelLikelihood trainer =
		// new CRFTrainerByL1LabelLikelihood(crf, 0.75);

		//trainer.addEvaluator(new PerClassAccuracyEvaluator(trainingInstances, "training"));
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
