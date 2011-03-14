package AII;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.regex.Pattern;

import cc.mallet.fst.HMM;
import cc.mallet.fst.HMMTrainerByLikelihood;
import cc.mallet.fst.MEMM;
import cc.mallet.fst.MEMMTrainer;
import cc.mallet.fst.PerClassAccuracyEvaluator;
import cc.mallet.fst.Transducer;
import cc.mallet.fst.TransducerEvaluator;
import cc.mallet.pipe.Pipe;
import cc.mallet.pipe.SerialPipes;
import cc.mallet.pipe.SimpleTaggerSentence2TokenSequence;
import cc.mallet.pipe.TokenSequence2FeatureSequence;
import cc.mallet.pipe.TokenSequence2FeatureVectorSequence;
import cc.mallet.pipe.TokenSequenceLowercase;
import cc.mallet.pipe.iterator.LineGroupIterator;
import cc.mallet.pipe.tsf.OffsetConjunctions;
import cc.mallet.pipe.tsf.RegexMatches;
import cc.mallet.pipe.tsf.TokenFirstPosition;
import cc.mallet.types.Alphabet;
import cc.mallet.types.FeatureSequence;
import cc.mallet.types.FeatureVector;
import cc.mallet.types.FeatureVectorSequence;
import cc.mallet.types.Instance;
import cc.mallet.types.InstanceList;
import cc.mallet.types.Sequence;
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

			FeatureSequence fs = new FeatureSequence(features);

			for (int l = 0; l < tokens.size(); l++) {
				fs.add(tokens.get(l).getText());
			}

			carrier.setData(fs);

			return carrier;
		}
	}

	public static HMM TrainHMM(String trainingFilename, int i, int p)
			throws IOException {

		ArrayList<Pipe> pipes = new ArrayList<Pipe>();
		pipes.add(new SimpleTaggerSentence2TokenSequence());
		pipes.add(new TokenSequenceLowercase());

		if (p == 0) {
			int[][] conjunctions = new int[1][];
			conjunctions[0] = new int[] { 1 };
			pipes.add(new OffsetConjunctions(conjunctions));
			pipes.add(new TokenSequence2FeatureSequence());
		} else if (p == 1) {
			int[][] conjunctions = new int[2][];
			conjunctions[0] = new int[] { 1 };
			conjunctions[1] = new int[] { -1 };
			pipes.add(new OffsetConjunctions(conjunctions));
			pipes.add(new TokenSequence2FeatureSequence());
		} else if (p == 2) {
			pipes.add(new HMMTrain.SimpleTokenSentence2FeatureSequence());
		}

		Pipe pipe = new SerialPipes(pipes);

		InstanceList trainingInstances = new InstanceList(pipe);
		trainingInstances.addThruPipe(new LineGroupIterator(new BufferedReader(
				new InputStreamReader(new FileInputStream(trainingFilename))),
				Pattern.compile("^\\s*$"), true));

		HMM hmm = new HMM(pipe, null);

		if (i == 0)
			hmm.addFullyConnectedStatesForBiLabels();
		else if (i == 1)
			hmm.addFullyConnectedStatesForLabels();
		else if (i == 2)
			hmm.addFullyConnectedStatesForThreeQuarterLabels(trainingInstances);
		else if (i == 3)
			hmm.addFullyConnectedStatesForTriLabels();
		else if (i == 4)
			hmm.addStatesForBiLabelsConnectedAsIn(trainingInstances);
		else if (i == 5)
			hmm.addStatesForHalfLabelsConnectedAsIn(trainingInstances);
		else if (i == 6)
			hmm.addStatesForLabelsConnectedAsIn(trainingInstances);
		else if (i == 7)
			hmm.addStatesForThreeQuarterLabelsConnectedAsIn(trainingInstances);
		else if (i == 8) {
			int[] orders = { 1 };
			Pattern forbiddenPat = Pattern.compile("\\s");
			Pattern allowedPat = Pattern.compile(".*");

			String startName = hmm.addOrderNStates(trainingInstances, orders,
					null, "O", forbiddenPat, allowedPat, true);
			for (int s = 0; s < hmm.numStates(); s++)
				hmm.getState(s).setInitialWeight(Transducer.IMPOSSIBLE_WEIGHT);
			hmm.getState(startName).setInitialWeight(0.0);
		}

		HMMTrainerByLikelihood trainer = new HMMTrainerByLikelihood(hmm);

		trainer.train(trainingInstances, 500);

		return hmm;
	}

	public static void main(String[] args) throws Exception {
		{
			String train = "corpus/train_2.txt";

			for (int p = 0; p < 3; p++) {
				for (int i = 0; i < 9; i++) {
					String model = "corpus/hmm_" + i + "_" + p + ".model";

					HMM modelObj = TrainHMM(train, i, p);

					ObjectOutputStream s = new ObjectOutputStream(
							new FileOutputStream(model));
					s.writeObject(modelObj);
					s.close();
				}
			}
		}
	}
}
