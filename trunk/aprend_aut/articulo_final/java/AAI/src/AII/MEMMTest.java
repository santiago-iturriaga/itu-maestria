package AII;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.ObjectInputStream;
import java.util.regex.Pattern;

import cc.mallet.fst.CRF;
import cc.mallet.fst.HMM;
import cc.mallet.pipe.Pipe;
import cc.mallet.pipe.iterator.LineGroupIterator;
import cc.mallet.types.FeatureSequence;
import cc.mallet.types.InstanceList;
import cc.mallet.types.Sequence;

public class MEMMTest {

	/**
	 * @param args
	 * @throws IOException 
	 * @throws FileNotFoundException 
	 */
	public static void main(String[] args) throws Throwable {
		String testingFilename = "corpus/test_1.txt";
		String modelFilename = args[0]; //"corpus/hmm_1.model";

		ObjectInputStream s = new ObjectInputStream(new FileInputStream(
				modelFilename));
		HMM hmm = (HMM) s.readObject();
		s.close();
		
		Pipe pipe = hmm.getInputPipe();
		InstanceList testingInstances = new InstanceList(pipe);
		testingInstances.addThruPipe(new LineGroupIterator(new BufferedReader(
				new InputStreamReader(new FileInputStream(testingFilename))),
				Pattern.compile("^\\s*$"), true));
	
		for (int i = 0; i < testingInstances.size(); i++) {
			FeatureSequence input = (FeatureSequence) testingInstances.get(i).getData();
			int[] p = input.getFeatures();
					
			Sequence[] outputs;
			outputs = new Sequence[1];
			outputs[0] = hmm.transduce(input);

			int k = outputs.length;
			boolean error = false;
			for (int a = 0; a < k; a++) {
				if (outputs[a].size() != input.size()) {
					error = true;
				}
			}
			if (!error) {
				for (int j = 0; j < input.size(); j++) {
					StringBuffer buf = new StringBuffer();
					for (int a = 0; a < k; a++)
						buf.append(outputs[a].get(j).toString()).append(" ");

					System.out.println(buf.toString());
				}
				System.out.println();
			}
		}
	}

}
