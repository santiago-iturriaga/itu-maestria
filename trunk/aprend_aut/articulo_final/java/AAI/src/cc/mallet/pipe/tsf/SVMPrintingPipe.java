/* Copyright (C) 2003 Univ. of Massachusetts Amherst, Computer Science Dept.
   This file is part of "MALLET" (MAchine Learning for LanguagE Toolkit).
   http://www.cs.umass.edu/~mccallum/mallet
   This software is provided under the terms of the Common Public License,
   version 1.0, as published by http://www.opensource.org.  For further
   information, see the file `LICENSE' included with this distribution. */
package cc.mallet.pipe.tsf;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.PrintWriter;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;

import cc.mallet.pipe.Pipe;
import cc.mallet.types.*;
import cc.mallet.util.Maths;
import cc.mallet.util.PropertyList;

/**
 * Created: Jul 6, 2005
 * 
 * @author <A HREF="mailto:casutton@cs.umass.edu>casutton@cs.umass.edu</A>
 * @version $Id: SequencePrintingPipe.java,v 1.1 2007/10/22 21:37:58 mccallum
 *          Exp $
 */
public class SVMPrintingPipe extends Pipe implements Serializable {
	private String logFile;
	private PrintWriter writer;
	private int index;

	public SVMPrintingPipe() {
		super(new Alphabet(), null);
		
		this.index = 0;
	}

	public SVMPrintingPipe(Alphabet dataDict) {
		super(dataDict, null);
		
		this.index = 0;
	}
	
	public SVMPrintingPipe(String logFile) throws FileNotFoundException {
		super(new Alphabet(), null);

		this.index = 0;
		this.logFile = logFile;
		this.writer = new PrintWriter(logFile);
	}

	public Instance pipe(Instance carrier) {
		Sequence data = (Sequence) carrier.getData();
		Sequence target = (Sequence) carrier.getTarget();

		if (data.size() != target.size())
			throw new IllegalArgumentException(
					"Trying to print into SimpleTagger format, where data and target lengths do not match\n"
							+ "data.length = "
							+ data.size()
							+ ", target.length = " + target.size());

		int N = data.size();

		if (data instanceof TokenSequence) {
			TokenSequence tokens = (TokenSequence) data;
			Alphabet features = getDataAlphabet();

			index++;
			for (int l = 0; l < tokens.size(); l++) {
				String currentTarget = target.get(l).toString();
				if (currentTarget.equals("O")) {
					writer.print("1 ");
				} else if (currentTarget.equals("SIN_TILDE")) {
					writer.print("2 ");
				} else if (currentTarget.equals("CON_TILDE")) {
					writer.print("3 ");
				}

				writer.print("qid:" + index + " ");

				int featureIndex;
				if (tokens.get(l).getFeatures() != null) {
					cc.mallet.util.PropertyList.Iterator iter = tokens.get(l)
							.getFeatures().iterator();

					ArrayList<Integer> features_aux = new ArrayList<Integer>();
					iter = tokens.get(l).getFeatures().iterator();
					while (iter.hasNext()) {
						iter.next();

						featureIndex = features.lookupIndex(iter.getKey());
						if (featureIndex >= 0) {
							features_aux.add(featureIndex);
						}
					}

					Collections.sort(features_aux);

					for (int i = 0; i < features_aux.size(); i++) {
						writer.print((features_aux.get(i) + 1) + ":1 ");
					}

					writer.println();

					// int wordPos = 0;
					// int currentPos = 0;
					// boolean encontrado = false;
					// while ((iter.hasNext()) && (!encontrado)) {
					// iter.next();
					//
					// if (iter.hasNext()) {
					// if (iter.getKey().startsWith("WORD=")) {
					// featureIndex = features.lookupIndex(iter
					// .getKey());
					// if (featureIndex >= 0) {
					// writer.print(featureIndex + ":1 ");
					// // + "("+ iter.getKey() + ")  ");
					// encontrado = true;
					// wordPos = currentPos;
					// }
					// }
					// currentPos++;
					// }
					// }
					//
					// iter = tokens.get(l).getFeatures().iterator();
					// currentPos = 0;
					// while (iter.hasNext()) {
					// iter.next();
					//
					// if (iter.hasNext()) {
					// if (wordPos != currentPos) {
					// featureIndex = features.lookupIndex(iter
					// .getKey());
					// if (featureIndex >= 0) {
					// writer.print(featureIndex + ":1 ");
					// // + "("+ iter.getKey() + ")  ");
					// }
					// }
					//
					// currentPos++;
					// }
					// }
				}
			}
		} else if (data instanceof FeatureVectorSequence) {
			throw new UnsupportedOperationException("Not yet implemented.");

			// FeatureVectorSequence fvs = (FeatureVectorSequence) data;
			// Alphabet dict = (fvs.size() > 0) ? fvs.getFeatureVector(0)
			// .getAlphabet() : null;
			//
			// for (int i = 0; i < N; i++) {
			// writer.print(dict.lookupObject(
			// fvs.getFeatureVector(i).indexAtLocation(0)).toString() + ' ');
			//
			// Object label = target.get(i);
			// writer.print(label);
			//
			// FeatureVector fv = fvs.getFeatureVector(i);
			// for (int loc = 1; loc < fv.numLocations(); loc++) {
			// writer.print(' ');
			// String fname = dict.lookupObject(fv.indexAtLocation(loc))
			// .toString();
			// double value = fv.valueAtLocation(loc);
			// if (!Maths.almostEquals(value, 1.0)) {
			// throw new IllegalArgumentException(
			// "Printing to SimpleTagger format: FeatureVector not binary at time slice "
			// + i + " fv:" + fv);
			// }
			// writer.print(fname);
			// }
			// writer.println();
			// }
		} else {
			throw new IllegalArgumentException(
					"Don't know how to print data of type " + data);
		}

		// writer.println();

		return carrier;
	}

	private void writeObject(ObjectOutputStream out) throws IOException {
		out.writeObject(logFile);
		
		writer.flush();
		writer.close();
	}

	private void readObject(ObjectInputStream in) throws IOException,
			ClassNotFoundException {
		logFile = (String) in.readObject();
		
		this.writer = new PrintWriter(logFile);
	}
}
