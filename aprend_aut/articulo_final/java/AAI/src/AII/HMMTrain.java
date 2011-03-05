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
import cc.mallet.types.InstanceList;

public class HMMTrain {
	public static HMM TrainHMM(String trainingFilename) throws IOException {
		ArrayList<Pipe> pipes = new ArrayList<Pipe>();

		int[][] conjunctions = new int[2][];
		conjunctions[0] = new int[] { -1 };
		conjunctions[1] = new int[] { 1 };
		
		pipes.add(new SimpleTaggerSentence2TokenSequence());
		pipes.add(new OffsetConjunctions(conjunctions));
		// pipes.add(new FeaturesInWindow("PREV-", -1, 1));
		// pipes.add(new TokenTextCharSuffix("C1=", 1));
		// pipes.add(new TokenTextCharSuffix("C2=", 2));
		// pipes.add(new TokenTextCharSuffix("C3=", 3));
		pipes.add(new RegexMatches("CAPITALIZED", Pattern.compile("^\\p{Lu}.*")));
		pipes.add(new RegexMatches("STARTSNUMBER", Pattern.compile("^[0-9].*")));
		pipes.add(new RegexMatches("HYPHENATED", Pattern.compile(".*[\\-|_].*")));
		// pipes.add(new RegexMatches("DOLLARSIGN", Pattern.compile(".*\\$.*")));
		pipes.add(new TokenFirstPosition("FIRSTTOKEN"));
		pipes.add(new TokenSequenceLowercase());
		pipes.add(new TokenSequence2FeatureSequence());

		Pipe pipe = new SerialPipes(pipes);

		InstanceList trainingInstances = new InstanceList(pipe);
		trainingInstances.addThruPipe(new LineGroupIterator(new BufferedReader(new InputStreamReader(new FileInputStream(trainingFilename))), Pattern.compile("^\\s*$"), true));
		
		HMM hmm = new HMM(pipe, null);

//		int[] orders = { 1 };
//	    Pattern forbiddenPat = Pattern.compile("\\s");
//	    Pattern allowedPat = Pattern.compile(".*");
		
//		String startName = hmm.addOrderNStates(trainingInstances, orders, null,
//				"O",forbiddenPat, allowedPat, true);
//		for (int i = 0; i < hmm.numStates(); i++)
//			hmm.getState(i).setInitialWeight(Transducer.IMPOSSIBLE_WEIGHT);
//		hmm.getState(startName).setInitialWeight(0.0);
		
		hmm.addStatesForLabelsConnectedAsIn(trainingInstances);
		
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
