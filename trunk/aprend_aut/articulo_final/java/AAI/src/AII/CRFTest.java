package AII;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.ObjectInputStream;
import java.io.PrintWriter;
import java.util.regex.Pattern;

import cc.mallet.fst.CRF;
import cc.mallet.fst.PerClassAccuracyEvaluator;
import cc.mallet.fst.TokenAccuracyEvaluator;
import cc.mallet.pipe.Pipe;
import cc.mallet.pipe.iterator.LineGroupIterator;
import cc.mallet.types.InstanceList;
import cc.mallet.types.Sequence;

public class CRFTest {

	public static void main(String[] args) throws Throwable {
		String testingFilename = "corpus/test_2.txt";
		String modelFilename = args[0]; //"corpus/crf_2.model";

		ObjectInputStream s = new ObjectInputStream(new FileInputStream(
				modelFilename));
		CRF crf = (CRF) s.readObject();
		s.close();
		
		Pipe pipe = crf.getInputPipe();
		
		InstanceList testingInstances = new InstanceList(pipe);
		testingInstances.addThruPipe(new LineGroupIterator(new BufferedReader(
				new InputStreamReader(new FileInputStream(testingFilename))),
				Pattern.compile("^\\s*$"), true));
		
//		PrintWriter o = new PrintWriter("log.alpha");
//		testingInstances.getAlphabet().dump(o);
		
		for (int i = 0; i < testingInstances.size(); i++) {
			Sequence input = (Sequence) testingInstances.get(i).getData();

			Sequence[] outputs;
			outputs = new Sequence[1];
			outputs[0] = crf.transduce(input);

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
